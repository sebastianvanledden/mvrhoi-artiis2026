#!/usr/bin/env python3
"""Preview or copy the MVRHOI overlay into a dedicated upstream checkout."""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Copy after checking all destinations')
    parser.add_argument('--overwrite', action='store_true', help='Allow replacement of different existing files')
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    required = ['common', 'modules/object_tracking/yolov5/train.py',
                'modules/object_tracking/yolov5/val.py', 'train_vidhoi.py',
                'eval_inference_vidhoi.py', 'eval_metrics_vidhoi.py']
    missing = [name for name in required if not (checkout / name).exists()]
    if missing:
        parser.error('Incomplete upstream checkout; missing: ' + ', '.join(missing))
    if checkout == ROOT or ROOT in checkout.parents and checkout != ROOT / 'upstream':
        parser.error('Use a separate upstream checkout or the ignored companion/upstream directory')

    overlay = ROOT / 'scripts/hoi-prediction-gaze-transformer'
    copies = [(p, checkout / p.relative_to(overlay)) for p in sorted(overlay.rglob('*'))
              if p.is_file() and '.DS_Store' != p.name and '__pycache__' not in p.parts]
    copies += [(p, checkout / 'configs' / p.name) for p in sorted((ROOT / 'configs').glob('*.yaml'))
               if p.name not in ('detector_opt.yaml', 'detector_hyp.yaml')]
    copies += [(p, checkout / 'mvrhoi_related' / p.name) for p in sorted((ROOT / 'data/mappings').glob('*.json'))]
    copies += [(p, checkout / 'vidhoi_related' / p.name) for p in sorted((ROOT / 'data/labels_vidhoi').glob('*.json'))]
    conflicts = []
    for source, target in copies:
        same = target.is_file() and source.read_bytes() == target.read_bytes()
        if target.exists() and not same:
            conflicts.append(target)
        action = 'unchanged' if same else 'replace' if target.exists() else 'copy'
        print(f'{action}: {source.relative_to(ROOT)} -> {target.relative_to(checkout)}')
    if not args.apply:
        print(f'Preview only: {len(copies)} files, {len(conflicts)} differing destinations. Use --apply to copy.')
        return
    if conflicts and not args.overwrite:
        parser.error('Different destination files exist. Review the preview and use --overwrite if intended; no files copied.')
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    print(f'Copied {len(copies)} files. Run pipeline commands from {checkout}.')
    print('Core upstream code and external weights have not been modified or validated.')


if __name__ == '__main__':
    main()
