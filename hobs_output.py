import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os


class HobsHeader(object):
    sim_head = '"SIMULATED EQUIVALENT"'
    obs_head = '"OBSERVED VALUE"'
    obs_name = '"OBSERVATION NAME"'
    date = 'DATE'
    dyear = 'DECIMAL_YEAR'

    header = {sim_head: None,
              obs_head: None,
              obs_name: None,
              date: None,
              dyear : None}


class HobsOut(dict):
    """
    Reads output data from Hobs file and prepares it for post processing.

    Class sets observations to an ordered dictionary based on observation name

    If observation name is consistant for a site, a time series is created
    for plotting!
    """

    def __init__(self, filename):
        super(HobsOut, self).__init__()
        self.name = filename

        self.__read_hobs_output()


    def __read_hobs_output(self):
        """
        Method to read a hobs output file. Dynamically sets header information
        and reads associated values.

        Sets values to HobsOut dictionary
        """

        with open(self.name) as hobout:
            for ix, line in enumerate(hobout):
                if ix == 0:
                    self.__set_header(line)

                else:
                    self.__set_dictionary_values(line)

    def __set_dictionary_values(self, line):
        """
        Method to set incoming hobs line to dictionary data values

        Args:
            line: (str)
        """
        t = line.strip().split()
        obsname = t[HobsHeader.header[HobsHeader.obs_name]]
        simval = float(t[HobsHeader.header[HobsHeader.sim_head]])
        obsval = float(t[HobsHeader.header[HobsHeader.obs_head]])
        residual = simval - obsval
        date = self.__set_datetime_object(t[HobsHeader.header[HobsHeader.date]])
        decimal_date = float(t[HobsHeader.header[HobsHeader.dyear]])

        if obsname in self:
            self[obsname]['simval'].append(simval)
            self[obsname]['obsval'].append(obsval)
            self[obsname]['date'].append(date)
            self[obsname]['decimal_date'].append(decimal_date)
            self[obsname]['residual'].append(residual)
        else:
            self[obsname] = {"simval": [simval], "obsval": [obsval],
                             "date": [date], "decimal_date": [decimal_date],
                             "residual": [residual]}

    def __set_header(self, line):
        """
        Reads header line and sets header index

        Args:
            line:

        Returns:

        """
        n = 0
        s = ""

        for i in line:
            s += (i)
            if s in HobsHeader.header:
                HobsHeader.header[s] = n
                n += 1
                s = ""

            elif s in (" ", "\t", "\n"):
                s = ""

            else:
                pass

        for key, value in HobsHeader.header.items():
            if value is None:
                raise AssertionError("HobsHeader headings must be updated")

    def __set_datetime_object(self, s):
        """
        Reformats a string of YYYY-mm-dd to a datetime object

        Args:
            s: (str) string of YYYY-mm-dd

        Returns:
            datetime.date
        """
        return dt.datetime.strptime(s, "%Y-%m-%d")

    def __get_date_string(self, date):
        """

        Args:
            datetime:

        Returns:

        """
        return date.strftime("%Y/%m/%d")

    def get_sum_squared_errors(self, obsname):
        """
        Returns the sum of squared errors from the residual

        Args:
            obsname: (str)

        Returns:
        """
        return sum([i**2 for i in self[obsname]['residual']])

    def get_rmse(self, obsname):
        """
        Returns the RMSE from the residual

        Args:
            obsname: (str)

        Returns:
        """
        return np.sqrt(np.mean([i**2 for i in self[obsname]['residual']]))

    def get_number_observations(self, obsname):
        """
        Returns the number of observations for an obsname

        Args:
            obsname: (str)

        Returns:
        """
        return len(self[obsname]['simval'])

    def get_maximum_residual(self, obsname):
        """
        Returns the datetime.date and maximum residual value

        Args:
            obsname: (str)

        Returns:
            tuple: (datetime.date, residual)
        """
        data = self[obsname]['residual']
        index = data.index(max(data))
        date = self[obsname]['date'][index]
        return (date, max(data))

    def get_minimum_residual(self, obsname):
        """
        Returns the datetime.date, minimum residual value

        Args:
            obsname: (str)

        Returns:
            tuple: (datetime.date, residual)
        """
        data = self[obsname]['residual']
        index = data.index(min(data))
        date = self[obsname]['date'][index]
        return (date, min(data))

    def get_mean_residual(self, obsname):
        """
        Returns the datetime.date, minimum residual value

        Args:
            obsname: (str)

        Returns:
            tuple: (datetime.date, residual)
        """
        data = self[obsname]['residual']
        return np.mean(data)

    def get_median_residual(self, obsname):
        """
        Returns the datetime.date, minimum residual value

        Args:
            obsname: (str)

        Returns:
            tuple: (datetime.date, residual)
        """
        data = self[obsname]['residual']
        return np.median(data)

    def get_maximum_residual_heads(self, obsname):
        """
        Returns the datetime.date, simulated, and observed
        heads at the maximum residual value
        Args:
            obsname: (str)

        Returns:
            tuple: (datetime.date, simulated head, observed head)
        """
        resid = self[obsname]['residual']
        index = resid.index(max(resid))
        observed = self[obsname]['obsval'][index]
        simulated = self[obsname]['simval'][index]
        date = self[obsname]['date'][index]
        return (date, simulated, observed)

    def get_minimum_residual_heads(self, obsname):
        """
        Returns the datetime.date, simulated, and observed
        heads at the maximum residual value
        Args:
            obsname: (str)

        Returns:
            tuple: (datetime.date, simulated head, observed head)
        """
        resid = self[obsname]['residual']
        index = resid.index(min(resid))
        observed = self[obsname]['obsval'][index]
        simulated = self[obsname]['simval'][index]
        date = self[obsname]['date'][index]
        return (date, simulated, observed)

    def get_residual_bias(self, filter=None):
        """
        Method to determine the bias of measurements +-
        by checking the residual. Returns fraction of residuals
        > 0.
        Args:
            filter: (str, list, tuple, or function)
                filtering criteria for writing statistics.
                Function must return True for filter out, false to use

        Returns:
            (float) fraction of residuals greater than zero
        """
        nobs = 0.
        ngreaterzero = 0.

        for obsname, meta_data in self.items():
            if self.__filter(obsname, filter):
                continue

            residual = np.array(meta_data['residual'])
            rgreaterzero = sum((residual > 0))

            nobs += residual.size
            ngreaterzero += rgreaterzero

        try:
            bias = ngreaterzero / nobs
        except ZeroDivisionError:
            raise ZeroDivisionError("No observations found!")

        return bias

    def write_dbf(self, dbfname, filter=None):
        """
        Method to write a dbf file from a the HOBS dictionary

        Args:
            dbfname: (str) dbf file name
        """
        import shapefile
        data = []
        for obsname, meta_data in self.items():

            if self.__filter(obsname, filter):
                continue

            for ix, val in enumerate(meta_data['simval']):
                data.append([obsname,
                             self.__get_date_string(meta_data['date'][ix]),
                             val,
                             meta_data['obsval'][ix],
                             meta_data['residual'][ix]])

        w = shapefile.Writer()
        w.field("HOBSNAME", fieldType="C")
        w.field("HobsDate", fieldType="D")
        w.field("HeadSim", fieldType='N', decimal=8)
        w.field("HeadObs", fieldType="N", decimal=8)
        w.field("Residual", fieldType="N", decimal=8)

        for rec in data:
            w.record(*rec)

        w.save(dbf=dbfname)

    def write_min_max_residual_dbf(self, dbfname, filter=None):
        """
        Method to write a dbf of transient observations
        using observation statistics

        Args:
            dbfname: (str) dbf file name

        """
        import shapefile
        data = []
        for obsname, meta_data in self.items():
            if self.__filter(obsname, filter):
                continue

            max_date, resid_max = self.get_maximum_residual(obsname)
            min_date, resid_min = self.get_minimum_residual(obsname)
            simval_max, obsval_max = self.get_maximum_residual_heads(obsname)[1:]
            simval_min, obsval_min = self.get_minimum_residual_heads(obsname)[1:]

            data.append([obsname,
                         self.get_number_observations(obsname),
                         self.__get_date_string(max_date), resid_max,
                         self.__get_date_string(min_date), resid_min,
                         simval_max, obsval_max, simval_min, obsval_min])

        w = shapefile.Writer()
        w.field("HOBSNAME", fieldType="C")
        w.field("FREQUENCY", fieldType="N")
        w.field("MaxDate", fieldType="C")
        w.field("MaxResid", fieldType='N', decimal=8)
        w.field("MinDate", fieldType="C", decimal=8)
        w.field("MinResid", fieldType="N", decimal=8)
        w.field("MaxHeadSim", fieldType="N", decimal=8)
        w.field("MaxHeadObs", fieldType="N", decimal=8)
        w.field("MinHeadSim", fieldType="N", decimal=8)
        w.field("MinHeadObs", fieldType="N", decimal=8)

        for rec in data:
            w.record(*rec)

        w.save(dbf=dbfname)

    def __filter(self, obsname, filter):
        """
        Boolean filetering method, checks if observation name
        is in the filter.

        Args:
            filter: (str, list, tuple, or function)
                filtering criteria for writing statistics.
                Function must return True for filter out, false for write to file

        Returns:
            bool: True if obsname in filter
        """
        if filter is None:
            return False

        elif isinstance(filter, list) or isinstance(filter, tuple):
            if obsname in list:
                return True

        elif isinstance(filter, str):
            if obsname == filter:
                return True

        elif callable(filter):
            if filter(obsname):
                return True

        else:
            raise Exception("Filter is not an appropriate type")

        return False

    def write_summary_statistics_csv(self, csvname, filter=None):
        """
        Method to write summary calibration statistics to a
        CSV file for analysis and reports

        Args:
            csvname: (str) csv file name
            filter: (str, list, tuple, or function)
                filtering criteria for writing statistics.
                Function must return True for filter out, false for write to file

        """
        data = []
        header = ["Well name", "Average", "Median",
                  "Minimum", "Maximum", "RMSE ft", "Frequency"]

        for obsname, meta_data in sorted(self.items()):

            if self.__filter(obsname, filter):
                continue

            resid_mean = self.get_mean_residual(obsname)
            resid_median = self.get_median_residual(obsname)
            resid_max = self.get_maximum_residual(obsname)[-1]
            resid_min = self.get_minimum_residual(obsname)[-1]
            rmse = self.get_rmse(obsname)
            frequency = self.get_number_observations(obsname)
            data.append((obsname, resid_mean, resid_median,
                         resid_min, resid_max, rmse, frequency))

        data = np.array(data, dtype=[('id', 'S15'), ('mean', float),
                                     ('med', float), ('min', float),
                                     ('max', float), ('rmse', float),
                                     ('num', np.int)])

        with open(csvname, "w") as foo:
            foo.write(",".join(header) + "\n")
            np.savetxt(foo, data, fmt="%15s,%.2f,%2f,%2f,%2f,%2f,%d")

    def plot(self, obsname, *args, **kwargs):
        """
        Plotting functionality from the hobs dictionary

        Args:
            obsname: (str) hobs package observation name
            *args: matplotlib args
            **kwargs: matplotlib kwargs

        """

        if obsname not in self:
            raise AssertionError("Obsname {}: not valid".format(obsname))

        axes = False
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
            axes = True

        obsval = self[obsname]['obsval']
        simval = self[obsname]['simval']
        date = self[obsname]['date']

        kwargs['label'] = "Observed WL"
        kwargs['color'] = 'r'
        if axes:
            ax.plot(date, obsval, *args, **kwargs)
        else:
            plt.plot(date, obsval, *args, **kwargs)

        kwargs['label'] = "Simulated Head"
        kwargs['color'] = 'b'
        if axes:
            ax.plot(date, simval, *args, **kwargs)
        else:
            plt.plot(date, simval, *args, **kwargs)

        if axes:
            return ax

    def plot_measured_vs_simulated(self, filter=None, **kwargs):
        """
        Plots measured vs. simulated data along a 1:1 profile.

        Args:
            filter: (str, list, tuple, or function)
                filtering criteria for writing statistics.
                Function must return True for filter out, false for write to file
            **kwargs: matplotlib.pyplot plotting kwargs

        Returns:
            axes: matplotlib axes object
        """
        axes = plt.subplot(111)

        for obsname, meta_data in self.items():

            if self.__filter(obsname, filter):
                continue

            simulated = meta_data['simval']
            observed = meta_data['obsval']

            axes.plot(observed, simulated, 'bo', markeredgecolor='k')


        return axes

    def plot_simulated_vs_residual(self, filter=None,
                                   histogram=False, **kwargs):
        """
        Creates a matplotlib plot of simulated heads vs residual

        Args:
            filter: (str, list, tuple, or function)
                filtering criteria for writing statistics.
                Function must return True for filter out, false for write to file
            histogram: (bool)
                Boolean variable that defines either a scatter plot (False)
                or a histogram (True) of residuals

            **kwargs: matplotlib.pyplot plotting kwargs

        Returns:
            axes: matplotlib axes object
        """
        axes = plt.subplot(111)

        if not histogram:
            for obsname, meta_data in self.items():

                if self.__filter(obsname, filter):
                    continue

                residual = meta_data['residual']
                observed = meta_data['obsval']

                axes.plot(observed, residual, 'bo', markeredgecolor="k")

        else:
            bins = np.arange(-25, 26, 5)

            d = {}
            for ix, bin in enumerate(bins):
                frequency = 0
                for obsname, meta_data in self.items():
                    if self.__filter(obsname, filter):
                        continue

                    for residual in meta_data['residual']:
                        if ix == 0:
                            if residual < bin:
                                frequency += 1

                        elif ix == (len(bins) - 1):
                            if residual > bin:
                                frequency += 1

                        else:
                            if bins[ix - 1] <= residual < bin:
                                frequency += 1

                if ix == 0:
                    name = "Less than {}".format(bin)

                elif ix == (len(bins) - 1):
                    name = "Greater than {}".format(bin)

                else:
                    name = "{} to {}".format(bins[ix - 1] + 1, bin)

                d[ix + 1] = {'name': name,
                             'frequency': frequency}

            tick_num = []
            tick_name = []
            for index, meta_data in sorted(d.items()):
                axes.bar(index, meta_data['frequency'], width=0.8)
                tick_num.append(index)
                tick_name.append(meta_data['name'])

            plt.xticks(tick_num, tick_name, rotation=45, fontsize=10)
            plt.xlim([0.5, len(tick_num) + 1])
            plt.subplots_adjust(left=0.12, bottom=0.22,
                                right=0.90, top=0.90,
                                wspace=0.20, hspace=0.20)
            plt.ylabel("Frequency")

        return axes



if __name__ == "__main__":
    ws = r'C:\Users\jlarsen\Desktop\Lucerne\Lucerne_OWHM\V0_initial_from_MODOPTIM\output'
    hobsname = "hobs.out"

    tmp = HobsOut(os.path.join(ws, hobsname))
    tmp.plot("04N01W01R04S", "o-")
    plt.legend(loc=0, numpoints=1)
    plt.show()
    print('break')