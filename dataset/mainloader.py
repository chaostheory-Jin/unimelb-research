#!/usr/bin/env python3
# main.py

import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='下载和处理情感数据集')
    parser.add_argument('--dataset', type=str, choices=['msp', 'iemocap', 'crema-d', 'all'], default='all',
                        help='要处理的数据集 (msp, iemocap, crema-d, all)')
    parser.add_argument('--download_only', action='store_true', help='仅下载数据集')
    parser.add_argument('--process_only', action='store_true', help='仅处理数据集')
    args = parser.parse_args()
    
    base_dir = "/Users/jinhongyu/Documents/GitHub/unimelb-research/dataset"
    
    # 下载数据集
    if not args.process_only:
        if args.dataset in ['msp', 'all']:
            print("下载 MSP-Podcast 数据集...")
            subprocess.run(['python', 'download_msp_podcast.py'])
        
        if args.dataset in ['iemocap', 'all']:
            print("下载 IEMOCAP 数据集...")
            subprocess.run(['python', 'download_iemocap.py'])
            
        if args.dataset in ['crema-d', 'all']:
            print("下载 CREMA-D 数据集...")
            subprocess.run(['python', 'download_crema_d.py'])
    
    # 处理数据集
    if not args.download_only:
        if args.dataset in ['msp', 'all']:
            print("处理 MSP-Podcast 数据集...")
            subprocess.run(['python', 'process_msp_podcast.py'])
        
        if args.dataset in ['iemocap', 'all']:
            print("处理 IEMOCAP 数据集...")
            subprocess.run(['python', 'process_iemocap.py'])
            
        if args.dataset in ['crema-d', 'all']:
            print("处理 CREMA-D 数据集...")
            subprocess.run(['python', 'process_crema_d.py'])
    
    print("完成!")

if __name__ == "__main__":
    main()
