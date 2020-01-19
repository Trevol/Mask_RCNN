import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", required=False, default='', help="'vis' or other")
parser.add_argument("--lr", required=False, default=None, type=float, help="'vis' or other")
parser.add_argument("--max-sessions", required=False, default=500, type=int, help="training stop after max-sessions")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args.start, args.lr)
