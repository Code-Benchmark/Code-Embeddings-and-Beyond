"""Main entrypoint for crawler commands."""

import argparse
import codesource.commands as algorithms

def main():
    """Execute the code2ast commandline interface."""

    parser = argparse.ArgumentParser(
        description="Crawl data sources for Python scripts.",
    )

    parser.add_argument(
        '--source',
        type=str,
        default='java',
        help='Data source to download. Available options: codesource',
    )

    parser.add_argument(
        '--infile',
        type=str,
        default='../data/example.csv',
        help='File to store labeled syntax trees from the datasource'
    )

    parser.add_argument(
        '--outfile',
        type=str,
        default='../data/example.pkl',
        help='File to store labeled syntax trees from the datasource'
    )


    args = parser.parse_args()

    if args.source.lower() == 'poj':
        fetch_func = algorithms.ast4poj
    elif args.source.lower() == 'java':
        fetch_func = algorithms.ast4java
    else:
        raise Exception('Please provide a data source.')

    fetch_func(args.infile, args.outfile)


if __name__ == '__main__':
    main()
