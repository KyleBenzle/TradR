import math

from binance.client import Client

from config.config_paramter import API_KEY, API_SECRET
from trader_logger import logger


class BinanceTrader:
    """
    Manager for buy and sell market order from binance
    """

    def __init__(self):
        if not API_KEY or not API_SECRET:
            raise ValueError('The API key and API Secret is required')
        self._client = Client(api_key=API_KEY, api_secret=API_SECRET, tld='com')
        # self._client.API_URL = 'https://testnet.binance.vision/api'

    def sell_order(self, asset):
        """
        Market sell order for specific asset.
        :param asset: string asset symbol for example "BTC"
        :type asset:str
        """
        asset_symbol = f"{asset}USDT"
        balance = self._get_account_balance(asset)

        try:

            if balance > 0:
                info = self._client.get_symbol_info(asset_symbol)
                step_size = 0.0
                for f in info['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        step_size = float(f['stepSize'])
                precision = int(round(-math.log(step_size, 10), 0))
                # I moved this line above the rounding operation becusuce we need to account for bianace minimum notional value allowed for an order on a symbol.
                balance = balance * 0.95 # Bianace Commission

                quantity = float(round(balance, precision))

                print("sell", quantity, asset)

                logger.rootLogger.info("About to sell %s out of %s", str(quantity), str(balance))

                print("About to sell", str(quantity), " out of ", str(balance))

                response = self._client.order_market_sell(
                    symbol=asset_symbol,
                    quantity=quantity,
                    recvWindow=50000
                )

                print("market sold ", asset_symbol)

                status = str(response['status'])
                if status and status.lower() == 'FILLED'.lower():
                    order_id = response['orderId']
                    orig_qty = response['origQty']
                    executed_qty = response['executedQty']
                    logger.rootLogger.info(
                        "sell order for %s filled with the order id %s qty %s exec_qty %s",
                        asset, str(order_id), str(orig_qty), str(executed_qty),
                        exc_info=1)
                else:
                    logger.rootLogger.critical('sell order fail for %s with the amount %s',
                                               asset, balance, exc_info=1)

            else:
                raise ValueError(f'No sufficient balance for {asset}')

        except:
            print("something went wrong with", asset, "sell")
            logger.rootLogger.exception('Error sell order:')

    def buy_order(self, asset):
        """
        Market sell order for specific asset.
        :param asset: string asset symbol for example "BTC"
        :type asset:str
        :return:
        """

        try:

            balance = self._get_account_balance('USDT')

            

            if balance == 0.0:
                raise ValueError(f'No sufficient balance for {asset}')


            amount = (20 / 100) * balance

            if amount == 0:
                raise ValueError(f'When I take 20% of your account assets USDT I can\'t Buy {asset}')

            if amount < 10:
                amount = 10

            print('amt ok')



            symbol = f'{asset}USDT'
            info = self._client.get_symbol_info(symbol)
            precision = info["quotePrecision"]

            print('data load ok')


            if precision is None or precision == 0:
                precision = 6

            

            print('precision',precision)



            response = self._client.order_market_buy(
                symbol=symbol,
                quoteOrderQty=float(round(amount, precision)),
                recvWindow=50000
            )

            status = str(response['status'])

            print('status ok')
            if status and status.lower() == 'FILLED'.lower():
                order_id = response['orderId']
                orig_qty = response['origQty']
                executed_qty = response['executedQty']
                logger.rootLogger.info(
                    "buy order for %s filled with the order id %s qty %s exec_qty %s",
                    asset, str(order_id), str(orig_qty), str(executed_qty),
                    exc_info=1)


            else:
                logger.rootLogger.critical('buy order fail for %s with the amount %s',
                                           asset, balance, exc_info=1)

        except:
            print("something went wrong with", asset, "buy")
            logger.rootLogger.exception('Error buy order:')

    def _get_account_balance(self, asset):
        """
        Get the current account balance
        :param asset: string asset symbol for example "BTC"
        :type asset:str
        :return: 0 if no asset found or asset value float.
        """
        balance = self._client.get_asset_balance(asset=f"{asset}")
        logger.rootLogger.info("Account Balance ---> {}".format(balance))
        if balance is not None:
            return float(balance['free'])
        else:
            return 0
