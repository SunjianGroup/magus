import argparse, importlib, logging
# from magus.calculators import CALCULATOR_PLUGIN
from magus.logger import set_logger
from magus import __version__, __picture__
from ..reconstruct.entrypoints import rcs_interface


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
        help="tolerance for symmetry finding",
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
        "--need_sorted",
        default =True,
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
        default=100,
        help="number of show in screen",
    )
    parser_sum.add_argument(
        "-sb",
        "--sorted-by",
        type=str,
        nargs="+",
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
        "-v",
        "--var",
        action="store_true",
        help="use variable composition mode",
    )
    parser_sum.add_argument(
        "-b",
        "--boundary",
        nargs="+",
        default=[],
        help="in variable composition mode: add boundary",
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
        help="variable composition search",
    )
    parser_pre.add_argument(
        "-m",
        "--mol",
        action="store_true",
        help="molecule crystal search",
    )
    # calculate
    parser_calc = subparsers.add_parser(
        "calculate",
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
        "-o",
        "--output-file",
        type=str,
        default="out.traj",
        help="output traj file"
    )
    parser_calc.add_argument(
        "-p",
        "--pressure",
        type=int, 
        default=None,
        help="add pressure"
    )
    # generate
    parser_gen = subparsers.add_parser(
        "generate",
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
    parser_update = subparsers.add_parser(
        "update",
        help="update magus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_update.add_argument(
        "-u",
        "--user",
        action='store_true',
        default=False,
        help="add --user to pip install"
    )
    parser_update.add_argument(
        "-f",
        "--force",
        action='store_true',
        default=False,
        help="add --force-reinstall to pip install"
    )

    #The following one line is interface to surface reconstruction, feel free to delete if not needed ;P
    rcs_interface(subparsers)
    
    #for developers: mutation test
    parser_mutate = subparsers.add_parser(
        "mutate",
        help="mutation test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_mus = ['input_file', 'seed_file', 'output_file']
    arg_def = ['input.yaml', 'seed.traj', 'result']
    for i,key in enumerate(arg_mus):
        parser_mutate.add_argument("-"+key[0], "--"+key, type=str, default=arg_def[i], help=key)

    #from .mutate import _applied_operations_
    #for key in _applied_operations_:
    #    parser_mutate.add_argument("--"+key, action='store_true', default=False, help = "add option to use operation!")
    
    parsed_args = parser.parse_args()
    if parsed_args.command is None:
        print(__picture__)
        parser.print_help()
    return parsed_args


def main():
    args = parse_args()
    dict_args = vars(args)
    if args.command in ['search', 'calculate', 'generate']:
        set_logger(level=dict_args['log_level'], log_path=dict_args['log_path'])
        log = logging.getLogger(__name__)
        log.info(__picture__)
    if args.command:
        try:
            f = getattr(importlib.import_module('magus.entrypoints.{}'.format(args.command)), args.command)
        except:
            raise RuntimeError(f"unknown command {args.command}")
        f(**dict_args)

if __name__ == "__main__":
    main()
