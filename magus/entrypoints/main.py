import argparse, logging
from magus.entrypoints import *
from magus.calculators import calc_dict

def parse_args():

    parser = argparse.ArgumentParser(
        description="Magus: Machine learning And Graph theory assisted "
                    "Universal structure Searcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")
    # search
    parser_search = subparsers.add_parser(
        "search",
        help="search structures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_search.add_argument(
        "-i",
        "--input-file",
        type=str, 
        default="input.yaml",
        help="the input parameter file in yaml format"
    )
    parser_search.add_argument(
        "-l",
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="set verbosity level by strings: ERROR, WARNING, INFO and DEBUG",
    )
    parser_search.add_argument(
        "-m",
        "--use-ml",
        action="store_true",
        help="use ml to accelerate(?) the search",
    )
    parser_search.add_argument(
        "-r",
        "--restart",
        action="store_true",
        help="Restart the searching.",
    )
    # summary
    parser_sum = subparsers.add_parser(
        "summary",
        help="summary the results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_sum.add_argument(
        "filenames",
        nargs="+",
        help="file (or files) to summary",
    )
    parser_sum.add_argument(
        "-p",
        "--prec",
        type=float,
        default=0.1,
        help="prec to judge symmetry",
    )
    parser_sum.add_argument(
        "-r",
        "--reverse",
        action="store_true",
        help="whether to reverse sort",
    )
    parser_sum.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="whether to save POSCARS",
    )
    parser_sum.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=".",
        help="where to save POSCARS",
    )
    parser_sum.add_argument(
        "-n",
        "--show-number",
        type=int,
        default=20,
        help="number of show in screen",
    )
    parser_sum.add_argument(
        "-sb",
        "--sorted-by",
        type=str,
        default="enthalpy",
        help="sorted by which arg",
    )
    parser_sum.add_argument(
        "-a",
        "--add-features",
        type=str,
        nargs="+",
        default=[],
        help="the features to be added to the show features",
    )

    # clean
    parser_clean = subparsers.add_parser(
        "clean",
        help="clean the path",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_clean.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="rua!!!!",
    )
    # prepare
    parser_pre = subparsers.add_parser(
        "prepare",
        help="generate InputFold etc to prepare for the search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_pre.add_argument(
        "-c",
        "--calc-type",
        choices=calc_dict.keys(),
        default="vasp",
        help="",
    )
    parser_pre.add_argument(
        "-v",
        "--var",
        action="store_true",
        help="bian zu fen sou suo",
    )
    parser_pre.add_argument(
        "-m",
        "--mol",
        action="store_true",
        help="fen zi jing ti sou suo",
    )
    # calculate
    parser_calc = subparsers.add_parser(
        "calc",
        help="calculate many structures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_calc.add_argument(
        "filename",
        type=str,
        help="structures to relax",
    )
    parser_calc.add_argument(
        "-m",
        "--mode",
        choices=["scf", "relax"],
        default="relax",
        help="scf or relax",
    )
    parser_calc.add_argument(
        "-i",
        "--input-file",
        type=str, 
        default="input.yaml",
        help="the input parameter file in yaml format"
    )
    parser_calc.add_argument(
        "-p",
        "--pressure",
        type=int, 
        default=None,
        help="hehe"
    )
    # generate
    parser_gen = subparsers.add_parser(
        "gen",
        help="generate many structures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_gen.add_argument(
        "-i",
        "--input-file",
        type=str, 
        default="input.yaml",
        help="the input parameter file in yaml format"
    )
    parser_gen.add_argument(
        "-o",
        "--output-file",
        type=str, 
        default="gen.traj",
        help="where to save generated traj"
    )
    parser_gen.add_argument(
        "-n",
        "--number",
        type=int, 
        default=10,
        help="generate number"
    )
    parsed_args = parser.parse_args()
    if parsed_args.command is None:
        parser.print_help()
    return parsed_args


def main():
    args = parse_args()
    dict_args = vars(args)
    if args.command == "search":
        search(**dict_args)
    elif args.command == "clean":
        clean(**dict_args)
    elif args.command == "prepare":
        prepare(**dict_args)
    elif args.command == "summary":
        summary(**dict_args)
    elif args.command == "calc":
        calculate(**dict_args)
    elif args.command == "gen":
        generate(**dict_args)
    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")

if __name__ == "__main__":
    main()