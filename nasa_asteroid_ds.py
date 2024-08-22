"""
@Project: Maman15

@Description : This project analyzes NASA asteroid data from a CSV file to calculate characteristics such as
maximum magnitudes and distances from Earth. It includes visualizations like histograms and pie charts to represent data
distribution and hazardous proportions. Additionally, functions assess linear relationships between asteroid speed and
size. Exception handling is incorporated to ensure robust data processing.

@ID : 318736253
@Author: Bar Abramov
@semester : 24b
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def load_data(file_name):
    """
    Loads data from a CSV file into a NumPy array.
    @param
        file_name (str): The name of the CSV file to load.
    @return
        np.ndarray: A NumPy array containing the loaded data.
    """
    try:
        with open(file_name, 'r') as file:
            header_row = [word.strip() for word in file.readline().split(',')]
            headers = np.array(header_row, dtype=str).reshape(1, -1)
            data = np.genfromtxt(file, dtype=None, delimiter=',', encoding='utf-8')
            data_array = np.array(data.tolist(), dtype=object)
            result = np.vstack((headers, data_array))
            return result
    except FileNotFoundError:
        print(f"File {file_name} cannot be found")
    except OSError:
        print(f"Unable to open file {file_name}")


def scoping_data(data, names):
    """
    Removes specified columns from the dataset.
    @param
        data (np.ndarray): Input dataset as a NumPy array.
        names (list): List of column names to exclude.
    @return
        np.ndarray: Updated dataset with excluded columns.
    """
    headers = data[0]
    columns_to_keep = [i for i in range(len(headers)) if headers[i] not in names]
    scoped_data = data[:, columns_to_keep]
    return scoped_data


def _convert_name(name):
    """
    Converts column names by replacing spaces, dots, and parentheses.
    @param
        name (str): Column name to convert.
    @return
        str: Converted column name.
    """
    return name.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")


def mask_data(data):
    """
    Masks data based on the condition of year above 2000.
    @param
        data (np.ndarray): Input dataset as a NumPy array.
    @return
        np.ndarray: Masked dataset based on the year condition.
    """
    try:
        index_col = _find_column(data, 'Close Approach Date')
        date = data[1:, index_col].astype('datetime64')
    except ValueError as e:
        raise ValueError(f"Error parsing datetime strings: {e}")
    date_mask = date >= np.datetime64('2000-01-01')
    filtered_data = np.hstack(([True], date_mask))
    return data[filtered_data]


def data_details(data):
    """
    After Removes specified columns from the dataset,
    prints details about the dataset: number of rows, number of columns, and headers.
    @param
        data (np.ndarray): Input dataset as a NumPy array.
    """
    updated_array = scoping_data(data, ['Neo Reference ID', 'Orbiting Body', 'Equinox'])
    print(
        f'Rows number is (without the header): {updated_array.shape[0] - 1}.'
        f'\nColumns number is: {updated_array.shape[1]}.')
    print(f'The headers are:\n{updated_array[0]}')


def max_absolute_magnitude(data):
    """
    Finds the maximum absolute magnitude in the dataset and returns its corresponding asteroid name.
    @param
        data (np.ndarray): Input dataset as a NumPy array.
    @return
        tuple: Tuple containing asteroid name and maximum absolute magnitude.
    """
    index_col_value = _find_column(data, 'Absolute Magnitude')
    values_col = data[1:, index_col_value]
    biggest = data[1:, index_col_value].max()
    index_row_biggest = np.where(values_col == biggest)[0][0]
    output = (data[index_row_biggest + 1][_find_column(data, 'Name')], biggest)
    return output


def closest_to_earth(data):
    """
    Finds the asteroid closest to Earth based on 'Miss Dist.(kilometers)'.
    @param
        data (np.ndarray): Input dataset as a NumPy array.
    @return
        str: Name of the asteroid closest to Earth.
    """
    index_col_value = _find_column(data, 'Miss Dist.(kilometers)')
    values_col = data[1:, index_col_value]
    biggest = data[1:, index_col_value].min()
    index_row_biggest = np.where(values_col == biggest)[0][0]
    return data[index_row_biggest + 1][_find_column(data, 'Name')]


def _find_column(data, column_name):
    """
    Finds the index of a column in the dataset based on its name.
    @param
        data (np.ndarray): Input dataset as a NumPy array.
        column_name (str): Name of the column to find.
    @return
        int: Index of the column.
    """
    return np.where(data[0] == column_name)[0][0]


def common_orbit(data):
    """
    Counts the occurrences of each orbit ID and returns a dictionary.
    @param
        data (np.ndarray): Input dataset as a NumPy array.
    @return
        dict: Dictionary where keys are orbit IDs and values are counts of occurrences.
    """
    orbits = data[1:, _find_column(data, 'Orbit ID')]
    counter = {}
    for value in orbits:
        if counter.get(value) is None:
            counter[value] = 1
        else:
            counter[value] += 1
    return counter


def min_max_diameter(data):
    """
    Computes the average minimum and maximum diameters of asteroids.
    @param
        data (np.ndarray): Input dataset as a NumPy array.
    @return
        tuple: Tuple containing average minimum and maximum diameters.
    """
    value_min = data[1:, _find_column(data, 'Est Dia in KM(min)')].mean()
    value_max = data[1:, _find_column(data, 'Est Dia in KM(max)')].mean()
    output = (value_min, value_max)
    return output


def axis_title_formatting(x_label, y_label, title, title_color='skyblue', axis_color='black', labels_color='black',
                          axis_pad=5, title_pad=5):
    """
    Applies common formatting to plots.
    """
    plt.xlabel(x_label, fontsize=10, fontweight='bold', color=labels_color, labelpad=axis_pad)
    plt.ylabel(y_label, fontsize=10, fontweight='bold', color=labels_color, labelpad=axis_pad)
    plt.xticks(fontsize=10, fontweight='bold', color=axis_color)
    plt.yticks(fontsize=10, fontweight='bold', color=axis_color)
    plt.title(title, fontsize=13, fontweight='bold', color=title_color, pad=title_pad)
    plt.grid()
    plt.tight_layout()


def plt_hist_diameter(data):
    """
    Plots a histogram of average asteroid diameters.
    @param
        data (np.ndarray): Input dataset as a NumPy array.
    """
    value_min = data[1:, _find_column(data, 'Est Dia in KM(min)')].astype(float)
    value_max = data[1:, _find_column(data, 'Est Dia in KM(max)')].astype(float)
    mean_values = np.add(value_min, value_max) / 2.0
    min_max_tuple = min_max_diameter(data)
    plt.hist(mean_values, bins=10, range=(min_max_tuple[0], min_max_tuple[1]), color='#5499C7', edgecolor='#273746')
    axis_title_formatting('Average Diameter (km)', 'Number of Asteroids', 'Histogram of Average Diameters of Asteroids',
                          '#CD6155', '#273746', '#CD6155', 10, 10)
    plt.savefig('diameter_histogram.png')
    plt.show()


def plt_hist_common_orbit(data):
    """
    Plots a histogram of asteroid counts by minimum orbit intersection.
    @param
        data (np.ndarray): Input dataset as a NumPy array.
    """
    the_values = common_orbit(data)
    min_key = min(the_values.keys())
    max_key = max(the_values.keys())
    values = []
    for key, count in the_values.items():
        values.extend([key] * int(count))
    plt.hist(values, bins=6, range=(min_key, max_key), color='#5499C7', edgecolor='#273746')
    axis_title_formatting('Minimum Orbit Intersection', 'Number of Asteroids',
                          'Histogram of Asteroids by Minimum Orbit Intersection', '#CD6155', '#273746', '#CD6155', 10,
                          10)
    plt.savefig('common_orbit_histogram.png')
    plt.show()


def plt_pie_hazard(data):
    """
    Plots a pie chart showing the percentage of hazardous and non-hazardous asteroids.
    @param
        data (np.ndarray): Input dataset as a NumPy array.
    """
    hazard_data = data[1:, _find_column(data, 'Hazardous')]
    plt.pie([hazard_data.shape[0] - hazard_data.sum(), hazard_data.sum()], labels=['Non-Hazardous', 'Hazardous'],
            colors=['#ABEBC6', '#EC7063'], textprops=dict(color='#2C3E50', fontsize=12, fontweight='bold'),
            startangle=180, autopct='%1.1f%%')
    plt.title('Percentage of Hazardous and Non-Hazardous Asteroids', fontsize=13, fontweight='bold', color='#2C3E50',
              pad=10)
    plt.savefig('pie_hazard.png')
    plt.show()


def plt_linear_motion_magnitude(data):
    """
    Performs linear regression between Absolute Magnitude and Miles per hour, and plots the results if significant.
    @param
        data (np.ndarray): Input dataset as a NumPy array.
    """
    absolute_magnitude = data[1:, _find_column(data, 'Absolute Magnitude')].astype(float)
    mile_per_hour = data[1:, _find_column(data, 'Miles per hour')].astype(float)
    a, b, r_value, p_value, std_err = stats.linregress(absolute_magnitude, mile_per_hour)
    # r_value is the correlation coefficient.
    # its negative it means that as one variable increases, the other variable tends to decrease.
    if p_value < 0.05:
        plt.scatter(absolute_magnitude, mile_per_hour, color='#5499C7', label='Data Points')
        plt.plot(absolute_magnitude, a * absolute_magnitude + b, color='#CB4335',
                 label=f'Fitted line (r={r_value:.2f})')
        axis_title_formatting('Absolute Magnitude', 'Miles per hour',
                              'Linear Relationship between Absolute\n Magnitude and Miles per hour', '#CD6155',
                              '#273746', '#CD6155', 5, 8)
        plt.legend()
        plt.savefig('linear_motion_magnitude.png')
        plt.show()


def main():
    data = load_data('nasa.csv')
    if data is None:
        return

    print("Original data:")
    print(data)

    print("\nScoping data...")
    columns_to_exclude = ['Absolute Magnitude', 'Est Dia in KM(min)']
    scoped_data = scoping_data(data, columns_to_exclude)
    print(scoped_data)

    print("\nMasking data...")
    masked_data = mask_data(data)
    print(masked_data)

    print("\nData details:")
    data_details(masked_data)

    print("\nMax absolute magnitude:")
    max_absolute = max_absolute_magnitude(data)
    print(max_absolute)

    print("\nClosest to earth:")
    closet = closest_to_earth(data)
    print(closet)

    print("\nCommon orbit:")
    common = common_orbit(data)
    print(common)

    print("\nMin Max diameter:")
    diameter = min_max_diameter(data)
    print(diameter)

    print("\nDisplays the graph of 'plt_hist_diameter'...")
    plt_hist_diameter(data)

    print("\nDisplays the graph of 'plt_hist_common_orbit'...")
    plt_hist_common_orbit(data)

    print("\nDisplays the graph of 'plt_pie_hazard'...")
    plt_pie_hazard(data)

    print("\nDisplays the graph of 'plt_linear_motion_magnitude'...")
    plt_linear_motion_magnitude(data)


if __name__ == '__main__':
    main()
