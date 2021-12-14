import argparse, importlib
# from magus.calculators import CALCULATOR_PLUGIN
from magus.logger import set_logger
from magus import __version__, __picture__

def parse_args():
    parser = argparse.ArgumentParser(
        description="Magus: Machine learning And Graph theory assisted "
                    "Universal structure Searcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        help="print version",
        action='version', 
        version=__version__
    )
    parser_log = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_log.add_argument(
        "-ll",
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="set verbosity level by strings: ERROR, WARNING, INFO and DEBUG",
    )
    parser_log.add_argument(
        "-lp",
        "--log-path",
        type=str,
        default="log.txt",
        help="set log file to log messages to disk",
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")
    # search
    parser_search = subparsers.add_parser(
        "search",
        parents=[parser_log],
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
        "--need-sort",
        action="store_true",
        help="whether to sort",
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
        default="Default",
        help="sorted by which arg",
    )
    parser_sum.add_argument(
        "-rm",
        "--remove-features",
        type=str,
        nargs="+",
        default=[],
        help="the features to be removed from the show features",
    ) 
    parser_sum.add_argument(
        "-a",
        "--add-features",
        type=str,
        nargs="+",
        default=[],
        help="the features to be added to the show features",
    )
    parser_sum.add_argument(
        "-c",
        "--cluster",
        action="store_true",
        help="whether to summary clusters",
    )
    parser_sum.add_argument(
        "-v",
        "--var",
        action="store_true",
        help="bian zu fen",
    )
    parser_sum.add_argument(
        "-t",
        "--atoms-type",
        choices=["bulk", "cluster"],
        default="bulk",
        help="",
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
    # parser_pre.add_argument(
    #     "-c",
    #     "--calc-type",
    #     choices=CALCULATOR_PLUGIN.keys(),
    #     default="vasp",
    #     help="",
    # )
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
        parents=[parser_log],
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
        parents=[parser_log],
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
    # check full
    parser_checkpack = subparsers.add_parser(
        "checkpack",
        parents=[parser_log],
        help="check full",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_checkpack.add_argument(
        "tocheck",
        nargs='?',
        choices=["all", "calculators", "comparators", "fingerprints"],
        default="all",
        help="the package to check"
    )
    # do unit test
    parser_test = subparsers.add_parser(
        "test",
        help="do unit test of magus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_test.add_argument(
        "totest",
        nargs='?',
        default="*",
        help="the package to test"
    )
    #For reconstructions, get a slab
    parser_slab = subparsers.add_parser(
        "getslab",
        help="get the slab model used in rcs-magus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_slab.add_argument(
        "filename",
        type=str,
        default= 'Ref/layerslices.traj',
        help="traj of slab model, default is './Ref/layerslices.traj'",
    )
    #generation energy analizer, a quick version of summary
    parser_ana = subparsers.add_parser(
        "analyze",
        help="get energy tendency of evolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_ana.add_argument(
        "filename",
        type=str,
        default= 'results',
        help="dictionary of results",
    )
    #for developers: mutation test
    parser_mutate = subparsers.add_parser(
        "mutate",
        help="mutation test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_mus = ['input_file', 'seed_file', 'output_file']
    arg_def = ['input.yaml', 'seed.traj', 'result']
    for i,key in enumerate(arg_mus):
        parser_mutate.add_argument("-"+key[0], "--"+key, type=str, default=arg_def[i])

    # from .mutate import _applied_operations_
    # for key in _applied_operations_:
    #     parser_mutate.add_argument("--"+key, type=int, default=0)

    parsed_args = parser.parse_args()
    if parsed_args.command is None:
        print(__picture__)
        parser.print_help()
    return parsed_args


def main():
    args = parse_args()
    dict_args = vars(args)
    if args.command in ['search', 'calc', 'gen']:
        set_logger(level=dict_args['log_level'], log_path=dict_args['log_path'])
    if args.command:
        try:
            f = getattr(importlib.import_module('magus.entrypoints.{}'.format(args.command)), args.command)
        except:
            raise RuntimeError(f"unknown command {args.command}")
        f(**dict_args)

if __name__ == "__main__":
    main()
