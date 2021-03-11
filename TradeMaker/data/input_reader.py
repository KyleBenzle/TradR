import csv
import re


class InputReader:
    """
      Class to read the data from the csv file
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def read_file_content(self):
        """
        read the file content form specific path.
        :return: the data as dict ex. BTC=1
        :raise ValueError if the file is not properly formatted.
        """
        data = self._import_csv()
        if len(data) != 2:
            raise ValueError('File is not properly formatted.')

        currency_value_dict = dict(zip(data[0], data[1]))
        return currency_value_dict

    def _import_csv(self):
        """
        if the Encoding is not UTF-16 this is will case a problem for you.
        :return: 2D Array with the values in it.
        """
        data = []
        with open(self.file_path, 'rt', encoding='UTF-8') as currencies:
            reader = csv.reader(
                (re.sub('[^a-zA-Z0-9,\n]', '', line.replace('\0', '').replace('\u200f', '')) for line in currencies),
                delimiter=",")
            for row in reader:
                if any(row):
                    data.append(row)
        print(data)
        return [data[0], data[-1]]
