import sqlite3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
def create_32x32_array(coordinates):
    """
    Create a 32x32 array and set the corresponding (x, y) positions based on the given coordinates.
    
    Parameters:
        coordinates: List of tuples containing (x, y) coordinates.
        
    Returns:
        grid: 32x32 array with the corresponding (x, y) positions set to 1.
    """
    # Create a 32x32 array initialized with zeros
    grid = np.zeros((32, 32))

    # Set the corresponding (x, y) positions to 1
    for x, y in coordinates:
        grid[x,y] = 1

    return grid

def scale_to_0_31(data):
    """
    Scale a list of values to the integer range 0-31.
    
    Parameters:
        data: List of numerical values.
        
    Returns:
        scaled_data: List of values scaled to the integer range 0-31.
    """
    # Find the minimum and maximum values of the data
    min_val = min(data)
    max_val = max(data)
    
    # Scale the data to the range [0, 31]
    scaled_data = [(x - min_val) / (max_val - min_val) * 31 for x in data]

    # Round the scaled data to integers
    scaled_data = [int(round(x)) for x in scaled_data]

    return scaled_data

def integrate_angular_speed(angular_speed_data, dt):
    """
    Integrate angular speed data to obtain angular displacement.
    
    Parameters:
        angular_speed_data: 2D array of shape (N, 3) containing angular speed data for x, y, and z axes.
        dt: Time step between data points.
        
    Returns:
        angular_displacement: 2D array of shape (N, 3) containing integrated angular displacement data.
    """
    angular_displacement = np.cumsum(angular_speed_data * dt, axis=0)
    return angular_displacement

def plot_trajectory(angular_displacement):
    """
    Plot the trajectory of the angular displacement.
    
    Parameters:
        angular_displacement: 2D array of shape (N, 3) containing integrated angular displacement data.
    """
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.plot(angular_displacement[:,0], angular_displacement[:,1])
    
    # ax.set_xlabel('X Displacement')
    # ax.set_ylabel('Y Displacement')
    # ax.set_zlabel('Z Displacement')
    
    # plt.title('Angular Displacement Trajectory')
    
    # plt.show()


    # Extract individual axes
    aX = -angular_displacement[:, 0]
    aY = angular_displacement[:, 1]
    aZ = angular_displacement[:, 2]

    # Plot x-y panel
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.plot(aX, aY, color='blue')
    plt.xlabel('aX')
    plt.ylabel('aY')
    plt.title('aX vs aY')

    # Plot x-z panel
    plt.subplot(1, 3, 2)
    plt.plot(aX, aZ, color='red')
    plt.xlabel('aX')
    plt.ylabel('aZ')
    plt.title('aX vs aZ')

    # Plot y-z panel
    plt.subplot(1, 3, 3)
    plt.plot(aY, aZ, color='green')
    plt.xlabel('aY')
    plt.ylabel('aZ')
    plt.title('aY vs aZ')

    plt.tight_layout()
    plt.show()




def select_data_by_name(name_key):
    # Connect to the SQLite database
    conn = sqlite3.connect('MotionChar.db')  # Change 'your_database_name.db' to your actual database filename
    cursor = conn.cursor()

    # Execute the query to select data by name
    cursor.execute("SELECT * FROM GestureTable WHERE name = ?", (name_key,))

    # Fetch all rows that match the query
    rows = cursor.fetchall()

    index = 0
    # Print the selected data
    for row in rows:
        # print(row)
        label = row[0]
        tester = row[1]
        trial=row[2]
        count = row[3]
        data = row[5]
        data_array = np.frombuffer(data, dtype=np.float32)
        data_array = data_array.reshape(-1, 14)
        df = pd.DataFrame(data_array, columns=['timestamp', 'pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ', 'xX', 'xY', 'xZ', 'aX', 'aY', 'aZ'])
        # print(df)

        dt = 0.01  # Example time step
        angular_speed_data = df[['aX', 'aY', 'aZ']].values

        # Integrate angular speed data to obtain angular displacement
        angular_displacement = integrate_angular_speed(angular_speed_data, dt)

        # Plot the trajectory of the angular displacement
        # plot_trajectory(angular_displacement)
        xDis = -angular_displacement[:,0]
        yDis = angular_displacement[:,1]
        x =scale_to_0_31(xDis)
        y = scale_to_0_31(yDis)

        # Pair the data as (x, y)
        coordinates = list(zip(x, y))

        # Create a 32x32 array and set the corresponding (x, y) positions
        grid = create_32x32_array(coordinates)

        # print(grid)
        # plt.imshow(grid, cmap='binary', interpolation='nearest')
        # plt.title(label)
        # plt.colorbar()
        # plt.show()

        lowerLabel = label[6]
        # np.savetxt("./gen/"+lowerLabel+"_"+tester+"_"+str(trial)+".dat", grid, fmt='%d')
        np.savetxt("./gen/"+lowerLabel+"_"+str(index)+".dat", grid, fmt='%d', delimiter=',')
        index += 1
        pass


    # Close the cursor and connection
    cursor.close()
    conn.close()


# name_key = "lower_t"  # Change 'example_name' to the name you want to search for
# select_data_by_name(name_key)

for ascii_code in range(ord('a'), ord('z') + 1):
    letter = chr(ascii_code)
    name_key = "upper_"+letter.upper()
    print(name_key)
    select_data_by_name(name_key)