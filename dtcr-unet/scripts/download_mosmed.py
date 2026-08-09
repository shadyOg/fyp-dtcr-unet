"""Helper to download MosMed data via Kaggle CLI (for Colab).

Usage (Colab):
  # after uploading kaggle.json to ~/.kaggle/
  python scripts/download_mosmed.py --dest /content/mosmeddata
"""
import argparse
import os
import subprocess


def download(dest, dataset_slug='andrewmvd/mosmed-covid19-ct-scans'):
    os.makedirs(dest, exist_ok=True)
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset_slug, '-p', dest, '--unzip']
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', required=True, help='Destination folder to download and unzip dataset')
    parser.add_argument('--dataset', default='andrewmvd/mosmed-covid19-ct-scans', help='Kaggle dataset slug')
    args = parser.parse_args()
    download(args.dest, args.dataset)


if __name__ == '__main__':
    main()
