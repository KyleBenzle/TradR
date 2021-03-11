import os
import sys

from data.input_reader import InputReader
from trader.binance_trader import BinanceTrader
from trader_logger import logger


def main(csv_file_path):
    try:
        input_reader = InputReader(csv_file_path)
        data = input_reader.read_file_content()
        for asset, signal in data.items():
            if int(signal) == 1:  # buy signal
                trader = BinanceTrader()
                trader.buy_order(asset)
                # print(asset)
            elif int(signal) == 0:  # sell signal
                trader = BinanceTrader()
                trader.sell_order(asset)
                print(asset)
        sys.exit(0)
    except ValueError as e:
        logger.rootLogger.exception('Error in main:')
        print(f"The CVS file '{csv_file_path}' is not properly formatted.")
        sys.exit(2)
    except Exception as e:
        print("Error Happened", e)
        logger.rootLogger.exception('Error in main:')
        sys.exit(2)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] is None or not sys.argv[1]:
        print("Please specify the csv file path. ")
        print("You can run this script using 'python trade_script.py \"PATH_TO_CSV_FILE\"'")
        sys.exit(2)
    file_path = sys.argv[1]
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print(f"The file '{file_path}' you pacified dose not exist. ")
        sys.exit(2)

    main(file_path)
