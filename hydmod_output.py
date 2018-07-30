import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os

ITMUNI_LUT = {1: "seconds",
              2: "minutes",
              3: "hours",
              4: "days",
              5: "years"}

COLORS = {i + 1: val for i, val in enumerate(["g", "y", "c", "saddlebrown",
                                              "olivedrab", "m", "deeppink",
                                              "maroon", "silver", "k"])}


class BinaryData(object):
    """
    The BinaryData class is a class to that defines the data types for
    integer, floating point, and character data in MODFLOW binary
    files. The BinaryData class is the super class from which the
    specific derived classes are formed.  This class should not be
    instantiated directly.

    """

    def __init__(self):

        self.integer = np.int32
        self.integerbyte = self.integer(1).nbytes

        self.character = np.uint8
        self.textbyte = 1

        return

    def set_float(self, precision):
        self.precision = precision
        if precision.lower() == 'double':
            self.real = np.float64
            self.floattype = 'f8'
        else:
            self.real = np.float32
            self.floattype = 'f4'
        self.realbyte = self.real(1).nbytes
        return

    def read_text(self, nchar=20):
        textvalue = self._read_values(self.character, nchar).tostring()
        if not isinstance(textvalue, str):
            textvalue = textvalue.decode().strip()
        else:
            textvalue = textvalue.strip()
        return textvalue

    def read_integer(self):
        return self._read_values(self.integer, 1)[0]

    def read_real(self):
        return self._read_values(self.real, 1)[0]

    def read_record(self, count, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return self._read_values(dtype, count)

    def _read_values(self, dtype, count):
        return np.fromfile(self.file, dtype, count)


class HydModOut(BinaryData, dict):
    """
    A dictionary based Hydmod Binary observation reader.

    Allows for sorting and acquiring observations based on ARR type,
    location, layer, etc....

    Inheits functionality from the FlopyBinaryData class which is reused
    in this script as BinaryData(object)
    """
    def __init__(self, filename):

        super(HydModOut, self).__init__()

        self.name = filename
        self.file = open(filename, 'rb')
        self.nobs = self.read_integer()
        self.totim = []
        self.datetime = []

        self.__get_precision()

        self.itmuni = self.read_integer()

        ctime = self.read_text(nchar=4)

        # setup metadata for read and dictionary protocols
        # maybe move this to its own hidden method to reduce clutter
        self.__binary_hydlbl = []
        self.__hydlbl_metadata = []
        for ix in range(self.nobs):
            lbl = self.read_text()
            self.__set_dictionary(lbl)
            self.__binary_hydlbl.append(lbl)

        self.__build_dtype()
        self.__read_data()
        self.file.close()

    @property
    def hydlbl(self):
        """
        Returns a list of hydlbls, by using the built in dictionary keys method
        """
        return self.keys()

    def __set_dictionary(self, lbl):
        """
        Takes the hydmod output label and uses it to set a dictionary entry
        containing metadata for later use

        Args:
            lbl: (str) hydmod label

        """
        arr, intyp, layer, hydlbl = self.__split_label(lbl)

        # todo: maybe break this part out for better clarity
        self.__hydlbl_metadata.append((arr, intyp, layer, hydlbl))

        if hydlbl in self:
            self[hydlbl][layer] = {"arr": arr, "intyp": intyp, "data": []}
        else:
            self[hydlbl] = {layer: {"arr": arr, "intyp": intyp, "data": []}}

    def __set_data_to_dict(self, record):
        """
        Parses the record read in from the hydmod file and appends data to
        the appropriate dictionary location

        Args:
            record: numpy record of values
        """
        record = record[0]

        self.totim.append(float(record[0]))
        for ix, metadata in enumerate(self.__hydlbl_metadata):
            hydlbl = metadata[-1]
            klay = metadata[-2]
            value = record[ix + 1]

            self[hydlbl][klay]["data"].append(value)

    def __split_label(self, lbl):
        """
        Splits the binary hydlbl and returns the Hydmod Arr, Intyp, KLay, and Hydlbl
        values

        Args:
            lbl: (str) binary hydlbl

        Returns:
            Arr, Intyp, KLay, Hydlbl: (str, str, int, str)
        """
        arr = lbl[:2]
        intyp = lbl[2]
        layer = int(lbl[3:6])
        hydlbl = lbl[6:]
        return arr, intyp, layer, hydlbl

    def __get_precision(self):
        """
        Sets the binary utility precision of floating point values
        """
        precision = 'single'
        if self.nobs < 0:
            self.nobs = abs(self.nobs)
            precision = "double"
        self.set_float(precision)


    def __build_dtype(self):
        """
        Builds a data type record for the binary file reader to use in importing
        data
        """
        dtype = [('totim', self.floattype)]
        for site in self.__binary_hydlbl:
            if not isinstance(site, str):
                site_name = site.decode().strip()
            else:
                site_name = site.strip()
            dtype.append((site_name, self.floattype))
        self.dtype = np.dtype(dtype)

    def __read_data(self):
        """
        Reads data in from the binary file
        """
        while True:
            try:
                r = self.read_record(count=1)
                if r.size == 0:
                    break
                else:
                    self.__set_data_to_dict(record=r)
            except:
                break
        return

    def date_time(self, start_date="1-1-1971", itmuni="days"):
        """
        Creates a date-time object from the totim array, sets this
        list to self.datetime and returns it to the user

        Args:
            start_date: (str) start date for datetime object
            itmuni: (str or int) can be text or corresponding modflow
                itmunit number

        Returns:
            (list) datetime.datetime objects
        """
        mo, day, year = [int(i) for i in start_date.split('-')]
        start_date = dt.datetime(year, mo, day)

        if isinstance(itmuni, int):
            itmuni = ITMUNI_LUT[itmuni]

        itmuni = itmuni.lower()

        if itmuni in ("seconds", "s", "sec", "secs", "second"):
            self.datetime = [start_date + dt.timedelta(seconds=time) for
                             time in self.totim]
        elif itmuni in ("minutes", "m", "min", "mins", "minute"):
            self.datetime = [start_date + dt.timedelta(minutes=time) for
                             time in self.totim]
        elif itmuni in ("hours", "hour", "h", "hrs", "hr"):
            self.datetime = [start_date + dt.timedelta(hours=time) for
                             time in self.totim]
        elif itmuni in ("day", "d", "days"):
            self.datetime = [start_date + dt.timedelta(days=time) for
                             time in self.totim]
        elif itmuni in ("year", "y", "years", "yr", "yrs"):
            # todo: maybe find a more elegant solution, this is kind of hacky
            self.datetime = [start_date + dt.timedelta(days=time*365.25) for
                             time in self.totim]

        return self.datetime


    def plot(self, hydlbl, *args, **kwargs):
        """
        Returns a matplotlib axes object for layering in a plot.

        Uses datetime if the user has called the get_date_time
        method.

        Args:
            hydlbl: (str) hydmod label
            *args: (matplotlib args)
            **kwargs: (matplotlib kwargs) as well as a layer option
        """

        if hydlbl not in self:
            raise AssertionError("hydlbl {}: not valid".format(hydlbl))

        strip = False
        if "strip_initial" in kwargs:
            kwargs.pop("strip_initial")
            strip = True

        axes = False
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
            axes = True

        time = self.totim
        if self.datetime:
            time = self.datetime

        if "layer" not in kwargs:

            for lay, record in self[hydlbl].items():
                kwargs["color"] = COLORS[lay]
                kwargs['label'] = "Layer {} {}".format(lay, record['arr'])

                if axes:
                    if strip:
                        ax.plot(time[1:], record["data"][1:], *args, **kwargs)
                    else:
                        ax.plot(time, record["data"], *args, **kwargs)
                else:
                    if strip:
                        plt.plot(time[1:], record["data"][1:], *args, **kwargs)
                    else:
                        plt.plot(time, record["data"], *args, **kwargs)

        else:
            layer = kwargs.pop("layer")
            if axes:
                if strip:
                    ax.plot(time[1:], self[hydlbl][layer]['data'][1:],
                             *args, **kwargs)
                else:
                    ax.plot(time, self[hydlbl][layer]['data'], *args, **kwargs)
            else:
                if strip:
                    plt.plot(time[1:], self[hydlbl][layer]['data'][1:],
                             *args, **kwargs)
                else:
                    plt.plot(time, self[hydlbl][layer]['data'], *args, **kwargs)

        if axes:
            return ax


if __name__ == "__main__":
    ws = r'C:\Users\jlarsen\Desktop\Lucerne\Lucerne_OWHM\V0_initial_from_MODOPTIM\output'
    hydname = "hydmod.out"

    tmp = HydModOut(os.path.join(ws, hydname))
    tmp.date_time(start_date="12-31-1941")
    print('break')
    tmp.plot("04N01E05G02S", lw=2)
    plt.legend(loc=0)
    plt.ylim([2700, 2850])
    plt.show()