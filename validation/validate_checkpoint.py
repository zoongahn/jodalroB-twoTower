#!/usr/bin/env python3
"""
체크포인트(.pt) 파일 검증 도구

Usage:
    # 단일 체크포인트 검증
    python validation/validate_checkpoint.py output/models/20251118_071938/checkpoint_epoch_1.pt

    # 전체 모델 디렉토리 검증
    python validation/validate_checkpoint.py output/models/20251118_071938/

    # 상세 정보 출력
    python validation/validate_checkpoint.py output/models/20251118_071938/ --verbose

    # JSON 형식으로 출력
    python validation/validate_checkpoint.py output/models/20251118_071938/ --json
"""

import torch
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import sys


class CheckpointValidator:
    """체크포인트 검증 클래스"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def validate_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        단일 체크포인트 검증

        Args:
            checkpoint_path: 체크포인트 파일 경로

        Returns:
            검증 결과 딕셔너리
        """
        result = {
            'file': str(checkpoint_path),
            'valid': False,
            'error': None,
            'info': {},
            'issues': []
        }

        try:
            # 체크포인트 로드
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 기본 정보 수집
            result['info']['epoch'] = checkpoint.get('epoch', 'N/A')
            result['info']['loss'] = checkpoint.get('loss', 'N/A')

            if 'metrics' in checkpoint:
                result['info']['metrics'] = checkpoint['metrics']

            # Model state dict 검증
            model_validation = self._validate_state_dict(
                checkpoint.get('model_state_dict', {}),
                'model'
            )
            result['info']['model'] = model_validation

            # Optimizer state dict 검증
            if 'optimizer_state_dict' in checkpoint:
                optimizer_validation = self._validate_optimizer_state(
                    checkpoint['optimizer_state_dict']
                )
                result['info']['optimizer'] = optimizer_validation

            # 문제 수집
            if model_validation['nan_count'] > 0:
                result['issues'].append(f"Model에 {model_validation['nan_count']}개 파라미터 NaN 발견")
            if model_validation['inf_count'] > 0:
                result['issues'].append(f"Model에 {model_validation['inf_count']}개 파라미터 Inf 발견")
            if model_validation['suspicious_count'] > 0:
                result['issues'].append(f"Model에 {model_validation['suspicious_count']}개 극단값 발견")

            if 'optimizer' in result['info']:
                opt_val = result['info']['optimizer']
                if opt_val['nan_count'] > 0:
                    result['issues'].append(f"Optimizer에 {opt_val['nan_count']}개 NaN 발견")
                if opt_val['inf_count'] > 0:
                    result['issues'].append(f"Optimizer에 {opt_val['inf_count']}개 Inf 발견")

            # 최종 판정
            result['valid'] = len(result['issues']) == 0

        except Exception as e:
            result['error'] = str(e)
            result['issues'].append(f"로드 실패: {e}")

        return result

    def _validate_state_dict(self, state_dict: Dict, dict_type: str) -> Dict[str, Any]:
        """State dict 검증"""
        validation = {
            'total_params': 0,
            'nan_params': [],
            'inf_params': [],
            'suspicious_params': [],
            'nan_count': 0,
            'inf_count': 0,
            'suspicious_count': 0
        }

        for name, param in state_dict.items():
            if not isinstance(param, torch.Tensor):
                continue

            validation['total_params'] += 1

            has_nan = torch.isnan(param).any()
            has_inf = torch.isinf(param).any()

            if has_nan:
                nan_count = torch.isnan(param).sum().item()
                validation['nan_params'].append({
                    'name': name,
                    'shape': list(param.shape),
                    'count': nan_count
                })
                validation['nan_count'] += 1

            if has_inf:
                inf_count = torch.isinf(param).sum().item()
                validation['inf_params'].append({
                    'name': name,
                    'shape': list(param.shape),
                    'count': inf_count
                })
                validation['inf_count'] += 1

            # 극단값 체크 (abs > 1e6)
            if not has_nan and not has_inf:
                max_abs_val = param.abs().max().item()
                if max_abs_val > 1e6:
                    validation['suspicious_params'].append({
                        'name': name,
                        'shape': list(param.shape),
                        'max_abs': max_abs_val
                    })
                    validation['suspicious_count'] += 1

        return validation

    def _validate_optimizer_state(self, opt_state_dict: Dict) -> Dict[str, Any]:
        """Optimizer state dict 검증"""
        validation = {
            'nan_count': 0,
            'inf_count': 0,
            'nan_params': [],
            'inf_params': []
        }

        if 'state' in opt_state_dict:
            for param_id, state in opt_state_dict['state'].items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if torch.isnan(value).any():
                            validation['nan_count'] += 1
                            validation['nan_params'].append({
                                'param_id': str(param_id),
                                'key': key
                            })
                        if torch.isinf(value).any():
                            validation['inf_count'] += 1
                            validation['inf_params'].append({
                                'param_id': str(param_id),
                                'key': key
                            })

        return validation

    def validate_directory(self, model_dir: Path) -> Dict[str, Any]:
        """
        모델 디렉토리 내 모든 체크포인트 검증

        Args:
            model_dir: 모델 디렉토리 경로

        Returns:
            전체 검증 결과
        """
        checkpoint_files = list(model_dir.glob('*.pt'))

        if not checkpoint_files:
            return {
                'directory': str(model_dir),
                'error': '체크포인트 파일(.pt)이 없습니다',
                'checkpoints': []
            }

        results = {
            'directory': str(model_dir),
            'total_checkpoints': len(checkpoint_files),
            'valid_count': 0,
            'invalid_count': 0,
            'checkpoints': []
        }

        for ckpt_path in sorted(checkpoint_files):
            validation = self.validate_checkpoint(ckpt_path)
            results['checkpoints'].append(validation)

            if validation['valid']:
                results['valid_count'] += 1
            else:
                results['invalid_count'] += 1

        return results

    def print_validation_result(self, result: Dict[str, Any], show_details: bool = False):
        """검증 결과 출력"""
        if 'directory' in result:
            # 디렉토리 전체 검증 결과
            self._print_directory_result(result, show_details)
        else:
            # 단일 체크포인트 검증 결과
            self._print_checkpoint_result(result, show_details)

    def _print_checkpoint_result(self, result: Dict[str, Any], show_details: bool):
        """단일 체크포인트 결과 출력"""
        print("=" * 80)
        print(f"체크포인트: {result['file']}")
        print("=" * 80)

        if result['error']:
            print(f"\n❌ 오류: {result['error']}\n")
            return

        # 기본 정보
        print(f"\n[기본 정보]")
        print(f"  Epoch: {result['info'].get('epoch', 'N/A')}")
        print(f"  Loss: {result['info'].get('loss', 'N/A')}")

        if 'metrics' in result['info']:
            print(f"\n[Metrics]")
            for key, value in result['info']['metrics'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")

        # Model 검증 결과
        model_info = result['info'].get('model', {})
        print(f"\n[Model State Dict]")
        print(f"  총 파라미터: {model_info.get('total_params', 0)}개")

        if model_info.get('nan_count', 0) > 0:
            print(f"  ❌ NaN: {model_info['nan_count']}개")
            if show_details and model_info.get('nan_params'):
                for param in model_info['nan_params'][:5]:
                    print(f"     - {param['name']}: {param['shape']}, count={param['count']}")
        else:
            print(f"  ✅ NaN 없음")

        if model_info.get('inf_count', 0) > 0:
            print(f"  ❌ Inf: {model_info['inf_count']}개")
            if show_details and model_info.get('inf_params'):
                for param in model_info['inf_params'][:5]:
                    print(f"     - {param['name']}: {param['shape']}, count={param['count']}")
        else:
            print(f"  ✅ Inf 없음")

        if model_info.get('suspicious_count', 0) > 0:
            print(f"  ⚠️ 극단값: {model_info['suspicious_count']}개 (|val| > 1e6)")
            if show_details and model_info.get('suspicious_params'):
                for param in model_info['suspicious_params'][:5]:
                    print(f"     - {param['name']}: {param['shape']}, max={param['max_abs']:.2e}")
        else:
            print(f"  ✅ 극단값 없음")

        # Optimizer 검증 결과
        if 'optimizer' in result['info']:
            opt_info = result['info']['optimizer']
            print(f"\n[Optimizer State Dict]")

            if opt_info.get('nan_count', 0) > 0:
                print(f"  ❌ NaN: {opt_info['nan_count']}개")
            else:
                print(f"  ✅ NaN 없음")

            if opt_info.get('inf_count', 0) > 0:
                print(f"  ❌ Inf: {opt_info['inf_count']}개")
            else:
                print(f"  ✅ Inf 없음")

        # 최종 판정
        print(f"\n[최종 판정]")
        if result['valid']:
            print(f"  ✅ 정상 - 사용 가능")
        else:
            print(f"  ❌ 손상됨 - 사용 불가")
            for issue in result['issues']:
                print(f"     - {issue}")

        print()

    def _print_directory_result(self, result: Dict[str, Any], show_details: bool):
        """디렉토리 전체 결과 출력"""
        print("=" * 80)
        print(f"디렉토리: {result['directory']}")
        print("=" * 80)

        if result.get('error'):
            print(f"\n❌ 오류: {result['error']}\n")
            return

        print(f"\n총 체크포인트: {result['total_checkpoints']}개")
        print(f"정상: {result['valid_count']}개")
        print(f"손상: {result['invalid_count']}개\n")

        for ckpt_result in result['checkpoints']:
            status = "✅" if ckpt_result['valid'] else "❌"
            filename = Path(ckpt_result['file']).name
            epoch = ckpt_result['info'].get('epoch', 'N/A')
            loss = ckpt_result['info'].get('loss', 'N/A')

            # Format epoch and loss
            epoch_str = str(epoch) if epoch != 'N/A' else 'N/A'
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) and loss != 'N/A' else str(loss)

            print(f"{status} {filename:30s} | Epoch: {epoch_str:>3s} | Loss: {loss_str}")

            if not ckpt_result['valid']:
                for issue in ckpt_result['issues']:
                    print(f"   → {issue}")

            if show_details:
                self._print_checkpoint_result(ckpt_result, show_details=True)

        print()


def main():
    parser = argparse.ArgumentParser(
        description='체크포인트(.pt) 파일 검증 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'path',
        type=str,
        help='체크포인트 파일 또는 모델 디렉토리 경로'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='상세 정보 출력'
    )

    parser.add_argument(
        '-j', '--json',
        action='store_true',
        help='JSON 형식으로 출력'
    )

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"❌ 오류: 경로를 찾을 수 없습니다: {path}")
        sys.exit(1)

    validator = CheckpointValidator(verbose=args.verbose)

    # 디렉토리 또는 파일 검증
    if path.is_dir():
        result = validator.validate_directory(path)
    else:
        result = validator.validate_checkpoint(path)

    # 결과 출력
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        validator.print_validation_result(result, show_details=args.verbose)

    # Exit code 설정
    if path.is_dir():
        sys.exit(0 if result['invalid_count'] == 0 else 1)
    else:
        sys.exit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
