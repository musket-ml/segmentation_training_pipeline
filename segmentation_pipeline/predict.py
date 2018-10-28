import argparse
from segmentation_pipeline import segmentation


def main():
    parser = argparse.ArgumentParser(description='Simple segmentation pipeline')
    parser.add_argument('--inputFolder',  type=str, required=True,
                        help='folder with iages that should be segmented')
    parser.add_argument('--output', type=str, required=True,
                        help='path to store segmentation maps')
    parser.add_argument('--config', type=str, default="",required=True,
                        help="path to experiment configuration file"
                        )

    args = parser.parse_args()
    cfg= segmentation.parse(args.config)
    cfg.predict_to_directory(args.inputFolder,args.output,batchSize=16)

if __name__ == '__main__':
    main()