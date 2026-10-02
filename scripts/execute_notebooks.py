"""Execute the three project notebooks in order using the active Python kernel."""
from pathlib import Path
import argparse
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))


def main():
    parser = argparse.ArgumentParser(description='DAY 1·DAY 2 노트북 순차 실행')
    parser.add_argument('--day', choices=['all', '1', '2'], default='all',
                        help='all: 전체, 1: EDA, 2: 피처 생성·모델링 (기본: all)')
    args = parser.parse_args()
    names = {
        'all': ['01_EDA.ipynb', '02_feature_engineering.ipynb', '03_modeling.ipynb'],
        '1': ['01_EDA.ipynb'],
        '2': ['02_feature_engineering.ipynb', '03_modeling.ipynb'],
    }[args.day]
    for name in names:
        path = ROOT / 'notebooks' / name
        print('Executing', name, flush=True)
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        try:
            NotebookClient(notebook, km=manager, timeout=600, resources={'metadata': {'path': str(ROOT)}}).execute()
        finally:
            nbformat.write(notebook, path)
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        print('Completed', name, flush=True)

if __name__ == '__main__':
    main()
