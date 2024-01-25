import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from wave_location import WaveLocationScatter
from location_data_scatter import WaveLocationScatterCollection
import ast
import resourcecode
import math
import folium
from streamlit_folium import st_folium


resourcecode.Client()
####################################################
####################################################

image_path = "Assets/Logo.png"  # Replace with the actual path to your image
st.set_page_config(layout="wide",page_title='Weptos research studio', page_icon = "Assets\mono.png")
st.image(image_path, width=300)

####################################################
####################################################

st.header('Weptos research studio')
st.write('Here You can analysis different locations with weptos different size of poroducts')
####################################################
####################################################












############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################



with st.expander("Wave Configuration "):
    st.subheader("Wave Configuration")
    st.write("In the Wave Configuration section, you can choose places from the list and check if they have enough power. You're in control here – add or remove locations easily. This way, the information stays up-to-date and fits what you need. It's a straightforward way to manage and understand power resources in different areas.")

    tab1, tab2, tab3 = st.tabs(["selection", "Location control ADD/Remove", "ResourceCode"])



    with tab1:

        loaded_collection = WaveLocationScatterCollection.load_from_json("Database/location_collection.json")

        if loaded_collection:
            st.subheader("Location Selection")

            # Extract location names for the selection box
            location_names = [location.location_name for location in loaded_collection.location_scatters]
            location_lat= [location.lat for location in loaded_collection.location_scatters]
            location_lng= [location.lng for location in loaded_collection.location_scatters]
            locations_df = pd.DataFrame({ "lat": location_lat, "lon": location_lng})
            locations_data=pd.DataFrame({ "location_name": location_names,"latitude": location_lat, "longitude": location_lng})
            # Create a selection box
            selected_location_name = st.selectbox("Choose a location", location_names)

            # Find the selected location
            selected_location = None
            for location in loaded_collection.location_scatters:
                if location.location_name == selected_location_name:
                    selected_location = location
                    break

        if selected_location:

            location_scatter=location.location_scatter
            (Hm0_start_value, Hm0_step_size, Tx_start_value, Tx_step_size) = (location.Hm0_start_value,location.Hm0_step_size, location.Tx_start_value, location.Tx_step_size)
            (Tm_array,Tp_array,Te_array,Hm0_array)=location.get_axis()

            col3,  col1 = st.columns([3,5])

            with col1:
                
                def display_wave_details():
                    col1, col2, col3, col4,col5 = st.columns([4,1,1,1,1])

                    col1.metric(label="Location Name", value=selected_location.location_name)
                    col2.metric(label="Latitude", value=str(selected_location.lat))
                    col3.metric(label="Longtitude", value=str(selected_location.lng))
                    col4.metric(label="Dataset Duration", value=str(selected_location.dataset_duration))
                    col5.metric(label="Water Depth", value=str(selected_location.water_depth))

                display_wave_details()

                def display_axis_details():
                    col1, col2, col3, col4 = st.columns([1,1,1,1])

                    col1.write(f"**Hm0 Start Value:** {Hm0_start_value}")
                    col2.write(f"**Hm0 Step Size:** {Hm0_step_size}")
                    col3.write(f"**Tx Start Value:** {Tx_start_value}")
                    col4.write(f"**Tx Step Size:** {Tx_step_size}")

            
                display_axis_details()            

                #scatter
                st.write("location scatter diagram")
                st.write(location_scatter)
                
            with col3:
                # map
                #st.map(locations_df)
                st.write("Available locations for analysis")
                from streamlit_folium import folium_static
                
                
                # Create a Folium map
                m1 = folium.Map(location=[selected_location.lat, selected_location.lng],
                                tiles= "CartoDB Positron",
                                zoom_start=4)
                
                # Add markers to the map
                for index,location in locations_data.iterrows():
                    folium.CircleMarker(
                        [location['latitude'], location['longitude']],
                        radius=5,  # Adjust the radius as needed
                        color='black',  # Change the color if desired
                        fill=True,
                        fill_color='black',
                        fill_opacity=1,
                        popup=location['location_name'],
                    ).add_to(m1)
                if selected_location:
                    folium.CircleMarker(
                        [selected_location.lat, selected_location.lng],
                        radius=6,  # Adjust the radius as needed
                        color='orange',  # Change the color if desired
                        fill=True,
                        fill_color='orange',
                        fill_opacity=1,
                        popup=location['location_name'],
                    ).add_to(m1)
                # Display the Folium map in Streamlit
                folium_static(m1)





        else:
            st.warning("No location selected.")
            
        

    with tab2:


        st.header("Location control")
        # Load WaveLocationScatterCollection from JSON file
        location_collection = WaveLocationScatterCollection.load_from_json("Database/location_collection.json")



        if location_collection:

            add_loc, rem_loc  = st.columns(2)
            with add_loc:
                st.subheader("Add a New Location")

                col5, col6,col10,col7, col8  = st.columns([2,1,1,1, 1])
                with col5:
                    i_location_name = st.text_input("Location Name")
                with col6:
                    i_lat = st.number_input("Latitude")
                with col10:
                    i_lng = st.number_input("Longtitute")
                with col7:
                    i_dataset_duration = st.number_input("Dataset Duration", value=50)
                with col8:
                    i_water_depth = st.number_input("Water Depth", value=25)

                # Input fields for adding a new location
                
                
                
                
                
                col1, col2,col3, col4  = st.columns(4)
                with col1:
                    i_Hm0_start_value = st.number_input("Hm0 Start Value", value=0.25)

                with col2:
                    i_Hm0_step_size = st.number_input("Hm0 Step Size", value=0.5)
                with col3:
                    i_Tx_start_value = st.number_input("Tx Start Value", value=0.5)
                with col4:
                    i_Tx_step_size = st.number_input("Tx Step Size", value=0.5)


                # Input field for scatter diagram data
                i_scatter_diagram_data = st.text_area("Enter Scatter Diagram Data (comma-separated values)", "")

                if st.button("Add Location"):
                    # Parse scatter diagram data from the input text area
                    i_location_scatter = np.array(ast.literal_eval(i_scatter_diagram_data))
                    # Create a new location object with scatter diagram data
                    i_new_location = WaveLocationScatter(
                        i_location_name,
                        i_lat,
                        i_lng,
                        i_dataset_duration,
                        i_water_depth,
                        i_Hm0_start_value,
                        i_Hm0_step_size,
                        i_Tx_start_value,
                        i_Tx_step_size
                    )

                    # Add the new location to the collection
                    location_collection.add_location_scatter(i_new_location)
                    i_new_location.location_scatter=i_location_scatter

                    # Save the updated collection to the JSON file
                    location_collection.save_to_json('Database/location_collection.json')
                    st.success(f"Location '{i_location_name}' added successfully!")



            with rem_loc:

                st.subheader("Remove a Location")

                # Extract location names for the selection box
                location_names = [location.location_name for location in location_collection.location_scatters]

                # Create a selection box for removing a location
                selected_location_name = st.selectbox("Choose a location to remove", location_names)

                if st.button("Remove Location"):
                    # Find the index of the selected location by its name
                    selected_location_index = next((i for i, location in enumerate(location_collection.location_scatters) if location.location_name == selected_location_name), None)

                    if selected_location_index is not None:
                        # Remove the selected location from the collection
                        removed_location_name = location_collection.location_scatters[selected_location_index].location_name
                        location_collection.location_scatters.pop(selected_location_index)

                        # Save the updated collection to the JSON file
                        location_collection.save_to_json('Database/location_collection.json')

                        st.success(f"Location '{removed_location_name}' removed successfully!")
                    else:
                        st.warning("Selected location not found!")



    with tab3:
        frac=st.slider("Fraction of Data of visible",
                  min_value=0.001, max_value=0.01, step=0.0005)
        data=resourcecode.get_grid_spec()
        sample_data=data.sample(frac=frac, random_state=42)
        col1, col2= st.columns(2)
        with col1:
            map_center = [56.481, 8.058]
            #my_map = folium.Map(location=map_center, zoom_start=12)
            m = folium.Map(location=map_center, zoom_start=4, tiles="CartoDB Positron")

            for index, row in sample_data.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,  # Adjust the radius as needed
                    color='transperent',  # Change the color if desired
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.2,
                    popup=f"Location: {row['name']}",
                ).add_to(m)    
    
            st_data = st_folium(m)

    


            # call to render Folium map in Streamlit
            

        with col2:
            
            if st_data['last_clicked'] is not None:
                closest_location = resourcecode.get_closest_station(latitude=st_data['last_clicked']['lat'], longitude=st_data['last_clicked']['lng'])
                pointId=resourcecode.data.get_closest_point(latitude=st_data['last_clicked']['lat'], longitude=st_data['last_clicked']['lng'])[0]
                # Find the corresponding location name in your DataFrame
                # Assuming the DataFrame has columns 'latitude', 'longitude', and 'name'
 
                closest_location_name = closest_location[0]
                closest_location = data[data['name'] == closest_location_name]
            
                # Display the result in Streamlit
                if not closest_location.empty:
                    closest_location_name = closest_location['name'].iloc[0]
                    st.write(f"Closest Station Name :  {closest_location_name}   \n Latitude : {closest_location['latitude'].iloc[0]}  \n Longitude : {closest_location['longitude'].iloc[0]} ")
                else:
                    st.write("No matching location found in the DataFrame.")
                    st.write("No matching location found in the DataFrame.")

                r_loc_name=st.text_input(label="Name of location ",value="LOCATION_NAME")
                r_location_name=r_loc_name+ "  " + closest_location['name'].iloc[0]
            
                if st.button("Add to the Data base"):
                    #confirmation = st.checkbox(f"Are you sure you want to add '{r_location_name}' to the database?")
                    #if confirmation:                    
                    def generate_scatter(pointId):
                        par1 = "hs"  # significant wave height
                        par2 = "t02" # wave period options: "t02" (mean period) "t0m1" (energy period) "t01" (mean period)
                        par3 = "dp"  # wave direction option: "dir" (mean direction) "dp" (direction at peak period)
                        par4 = "cge" # wave power
        
                        client=resourcecode.Client()
                        query_string = f"""
                        {{
                        "node": {pointId},
                        "parameter": ["{par1}", "{par2}", "{par3}", "{par4}"]
                        }}
                        """
                        data = client.get_dataframe_from_criteria(query_string)
        
                        grid_info = resourcecode.data.get_grid_field()
                        grid_info = grid_info.set_index(grid_info.columns[0])
                        # Print statistics
                        st.write(f"Hs Min:", data[par1].min())
                        st.write(f"Hs Max:", data[par1].max())
                        st.write(f"Tm Min:", data[par2].min())
                        st.write(f"Tm Max:", data[par2].max())
                        st.write("Pwave Mean:", data[par4].mean())

                        # Generate scatter diagram from data
                        #x = np.random.rand(10000)  # Example x-axis data
                        #y = np.random.rand(10000)  # Example y-axis data
                        x = data[par2]  # Example x-axis data
                        y = data[par1]  # Example y-axis data

                        x_max = math.ceil(data[par2].max() / 2) * 2
                        y_max = math.ceil(data[par1].max() / 2) * 2
        
                        # Define intervals for both axes
                        x_intervals = np.linspace(0, x_max, 1*x_max+1)  # Divide x-axis into n intervals
                        y_intervals = np.linspace(0, y_max, 2*y_max+1)  # Divide y-axis into n intervals
        
                        # Calculate bivariate probabilities
                        bivariate_probs = np.zeros((len(x_intervals) - 1, len(y_intervals) - 1))
        
                        for i in range(len(x_intervals) - 1):
                            for j in range(len(y_intervals) - 1):
                                x_in_interval = (x >= x_intervals[i]) & (x < x_intervals[i + 1])
                                y_in_interval = (y >= y_intervals[j]) & (y < y_intervals[j + 1])
                                bivariate_probs[i, j] = np.mean(x_in_interval & y_in_interval)
        

                        # Convert the bivariate_probs array to a DataFrame. x / y values set central in interval
                        df=[]
                        df = pd.DataFrame(bivariate_probs.T, index=(y_intervals[:-1]+(y_intervals[0]+y_intervals[1])/2), columns=x_intervals[:-1]+(x_intervals[0]+x_intervals[1])/2)
                        r_Hm0=(y_intervals[0]+y_intervals[1])/2
                        r_Tx=(x_intervals[0]+x_intervals[1])/2
                        return bivariate_probs.T,df,r_Hm0,r_Tx
                    
                    r_location_scatter,full_scat,r_Hm0,r_Tx=generate_scatter(pointId)
                    r_lat=closest_location['latitude'].iloc[0]
                    r_lng=closest_location['longitude'].iloc[0]
                    r_dataset_duration=25
                    r_water_depth=closest_location['depth'].iloc[0]
                    r_Hm0_start_value=r_Hm0
                    r_Hm0_step_size=0.5
                    r_Tx_start_value=r_Tx
                    r_Tx_step_size=1
                    st.write(r_Hm0,r_Tx)
                    st.write(full_scat)
                    
                        
                    r_location_scatter = np.array(r_location_scatter)
                    # Create a new location object with scatter diagram data
                    r_new_location = WaveLocationScatter(
                        r_location_name,
                        r_lat,
                        r_lng,
                        r_dataset_duration,
                        r_water_depth,
                        r_Hm0_start_value,
                        r_Hm0_step_size,
                        r_Tx_start_value,
                        r_Tx_step_size
                    )
    
                    # Add the new location to the collection
                    location_collection.add_location_scatter(r_new_location)
                    r_new_location.location_scatter=r_location_scatter
    
                    # Save the updated collection to the JSON file
                    location_collection.save_to_json('Database/location_collection.json')

                    st.success(f"Location '{r_location_name}' added successfully!")
                    # Export the DataFrame to an Excel file
                    #df.to_excel(excel_file_path)
                





    # Constants for power calculations
    rho = 1029  # density of seawater in kg/m^3
    g = 9.816    # acceleration due to gravity in m/s^2

    #matrix calculations

    # Normalize each element by the sum of all elements in the matrix
    prob_matrix = location_scatter / location_scatter.sum()

    # Calculate wave power using vectorized operations
    wave_power = (rho * g**2 / (64 * np.pi * 1000)) * np.outer(Hm0_array**2, Te_array)

    # Calculate wave contribution matrix (element-wise multiplication of probability matrix and wave power matrix)
    wave_contribution = prob_matrix * wave_power



    def plot_3d_matrix(matrix, title):
        fig = go.Figure(data=[go.Surface(z=matrix, x=Tm_array, y=Hm0_array, colorscale='Viridis')])
        fig.update_layout(
            title=title,
            margin=dict(l=20, r=20, b=20, t=20),
                                    height=450,
                                    autosize=False,
            scene=dict(
                

                xaxis_title='Tm',
                yaxis_title='Hm0',
                zaxis_title=title,
                
            )
        )
        st.plotly_chart(fig)

    st.subheader('Location Analysis')

    col1, col2, col3 = st.columns([1, 1,1])



    with col1:
    # Display 3D plots with different perspectives
        
        plot_3d_matrix(prob_matrix, 'Probability')
    with col2:

        plot_3d_matrix(wave_power, 'Wave Power (kW/m)')
    with col3:

        plot_3d_matrix(wave_contribution, 'Wave Contribution (kW/m)')


    avg_wave_power = wave_contribution.sum()
    st.metric(label=f"Average wave power in {selected_location.location_name}", value=str(round(avg_wave_power, 2))+" kW/m")
























############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
with st.expander("Device Configuration"):
    st.header("Device Configuration")
    st.write("In the Device Configuration section, you have the ability to manage specific features of your device, like adjusting the rotor size and generator capacity. This allows you to customize your device settings. Moreover, there are plans to introduce a feature where you can save particular configurations for convenience. This means you won't have to set everything each time in the future, making the process smoother and more efficient.")

    non_dim_curve = np.array([[2.000e-01, 0.000e+00],
        [1.100e+00, 1.000e-01],
        [3.280e+00, 4.760e-01],
        [3.820e+00, 5.700e-01],
        [4.300e+00, 4.910e-01],
        [4.950e+00, 4.330e-01],
        [8.480e+00, 2.250e-01],
        [1.241e+01, 1.150e-01],
        [1.560e+01, 6.400e-02],
        [2.280e+01, 3.800e-02],
        [3.137e+01, 2.000e-02],
        [7.935e+01, 0.000e+00]])

    def create_eff_func(non_dim_curve, rotor_size):
        L0m_over_R = non_dim_curve[:, 0]
        Eff = non_dim_curve[:, 1]

        # Calculate Torque (Tm)
        wave_lenght = 1.56
        Tm = (L0m_over_R * rotor_size / wave_lenght)**(1/2)

        # Calculate Efficiency
        Efficiency = Eff

        return Tm, Efficiency

    def find_efficiency(Tx, Tm, Efficiency):
        # Interpolate the efficiency for the given Tx
        efficiency = np.interp(Tx, Tm, Efficiency)
        
        return efficiency

    def create_eff_matrix(Tm_array, Tm, Hm0_array, Efficiency):
        # Create empty matrix to store efficiency values
        eff_matrix = np.zeros(( len(Hm0_array),len(Tm_array)))

        for i, Hm0_val in enumerate(Hm0_array):
            for j, Tm_val in enumerate(Tm_array):
                eff_matrix[i, j] = find_efficiency(Tm_val, Tm, Efficiency)

        return eff_matrix
    
    

    
    config, plot_col,plot2_col = st.columns([3, 1,1])
    with config:
        co1, co2 = st.columns([6, 1])

        with co1:
            
            rotor_size = st.slider("Select a rotor size between 0.1 and 10 (m)", 0.1, 50.0, step=0.1,value=4.5)
        with co2:
            num_of_rotor = st.number_input("Select a number rotors", 10, 25, step=1,value=24)

        install_capacity = st.slider("Select a generator capacity between 0.1 and 5000 (kW)", 100, 10000, step=50,value=1000)

        active_width=rotor_size*1.2*num_of_rotor
        Tm, Efficiency = create_eff_func(non_dim_curve, rotor_size)

        def display_divice_config():
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(label="Rotor size", value=str(round(rotor_size, 2))+" m")
            col2.metric(label="Number of Rotors", value=str(num_of_rotor))
            col3.metric(label="Installed Capacity", value=str(round((install_capacity), 2))+" MW")
            col4.metric(label="Active width", value=str(round((active_width), 2))+" m")

            
        display_divice_config()    
                
    with plot_col:
        
        # Separate x and y data
        x_values = non_dim_curve[:, 0]
        y_values = non_dim_curve[:, 1]

        # Plotting the graph using Streamlit
        # Create a DataFrame for Streamlit
        # Separate x and y data
        x_values = non_dim_curve[:, 0]
        y_values = non_dim_curve[:, 1]

        # Plotting the graph using Matplotlib
        fig, ax = plt.subplots()
        ax.plot(x_values, y_values)
        ax.set_title('Non-Dimensional Curve')
        ax.set_xlabel('L0m/øR [ - ]')  # X-axis label
        ax.set_ylabel('Efficiency')     # Y-axis label

        # Display the plot in Streamlit
        st.pyplot(fig)

    with plot2_col:
                # Separate x and y data
        x_values = np.arange(1,25.1,0.1)
        y_values = find_efficiency(x_values, Tm, Efficiency)

        # Find the index of the peak value
        peak_index = np.argmax(y_values)
        peak_value = y_values[peak_index]
        corresponding_Tm = x_values[peak_index]

        # Plotting the graph using Matplotlib
        fig, ax = plt.subplots()
        ax.plot(x_values, y_values)

        # Highlight the peak point
        ax.plot(corresponding_Tm, peak_value,  label='Peak')

        # Annotate the corresponding Tm value
        ax.annotate(f'Tm={corresponding_Tm:.2f}', xy=(corresponding_Tm, peak_value), xytext=(corresponding_Tm+5, peak_value - 0.05),
                    arrowprops=dict(facecolor='black', arrowstyle='-'))

        ax.set_title('Dimensional Curve')
        ax.set_xlabel('Tm (seconds)')  # X-axis label
        ax.set_ylabel('Efficiency')     # Y-axis label

        # Display the plot in Streamlit
        st.pyplot(fig)

   # st.write("### Rotor size:", rotor_size,"Number of Rotors:", num_of_rotor,"Installed Capacity:", install_capacity)


    #Tm_array = np.array([3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5])
    #Hm0_array = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.])





























############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

with st.expander("Power Production"):
    st.subheader("Power Production")
    st.write("In the Power Production section, you can view the power output corresponding to the wave configuration and device settings you selected earlier. This section directly communicates with the Device Configuration, providing real-time updates on power production based on your chosen parameters. It's an interactive feature that gives you immediate insights into how your device configuration influences actual power generation.")








  
    # Create the  matrix
    Tm, Efficiency = create_eff_func(non_dim_curve, rotor_size)

    eff_matrix = create_eff_matrix(Tm_array, Tm, Hm0_array, Efficiency)

    power_matrix=np.clip(wave_power*active_width*eff_matrix, a_min=None, a_max=install_capacity)

    power_contribution=prob_matrix*power_matrix

    
    col1, col2, col3 = st.columns([1, 1,1])



    with col1:
    # Display 3D plots with different perspectives
        
        plot_3d_matrix(eff_matrix,"Efficiency metrics")
    with col2:

        plot_3d_matrix(power_matrix, 'Power matrix (kW)')
    with col3:

        plot_3d_matrix(power_contribution, 'Power Contribution (kW)')




    average_generatable_power=power_contribution.sum()
    anual_production=average_generatable_power*365.25*24/1000
    capacity_factor=average_generatable_power/install_capacity
    overall_eff         = average_generatable_power/(avg_wave_power*active_width)




    def display_power_prod():
        col1, col2, col3, col4,col5, col6 = st.columns(6)

        col1.metric(label="Average wave power", value=str(round(avg_wave_power, 2))+" kW/m")
        col2.metric(label="Average power generation", value=str(round(power_contribution.sum(), 2))+" kW   ")
        col3.metric(label="Anual energy production", value=str(round((average_generatable_power*365*24/1000), 2))+" MWh")
        col4.metric(label="Capacity factor", value=round((capacity_factor), 2))
        col5.metric(label="Overall Effficiency", value=str(round((overall_eff*100), 1))+"%")
        col6.metric(label="Active width", value=round((active_width), 1))

        
    display_power_prod()












############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
with st.expander("Additional analysis"):
    st.subheader("Additional analysis")
    st.write("Here you can generate additional 3D plots to analysis how the production depends on device parameter")



    def gen_end(non_dim_curve,Tm_array,Hm0_array,wave_power,prob_matrix):
    # Input parameters

        rotor_size_range = np.arange(1, 20.2, 0.2)
        install_capacity_range = np.arange(250, 20050, 250)
        num_of_rotor=24

        X, Y = np.meshgrid(rotor_size_range, install_capacity_range)
        average_generatable_power_matrix = np.zeros_like(X)
        anual_production_matrix = np.zeros_like(X)
        capacity_factor_matrix = np.zeros_like(X)
        overall_eff_matrix = np.zeros_like(X)

        progress_bar = st.progress(0)
        total_iterations = len(rotor_size_range) * len(install_capacity_range)
        current_iteration = 0

        for i in range(len( install_capacity_range)):
            for j in range(len(rotor_size_range)):
                rotor_size = X[i, j]
                install_capacity = Y[i, j]
                active_width=rotor_size*1.1*num_of_rotor

            # Create the  matrix
                Tm, Efficiency = create_eff_func(non_dim_curve, rotor_size)

                eff_matrix = create_eff_matrix(Tm_array, Tm, Hm0_array, Efficiency)

                power_matrix=np.clip(wave_power*active_width*eff_matrix, a_min=None, a_max=install_capacity)

                power_contribution=prob_matrix*power_matrix
                average_generatable_power=power_contribution.sum()
                anual_production=average_generatable_power*365.25*24/1000
                capacity_factor=average_generatable_power/install_capacity
                overall_eff         = average_generatable_power/(avg_wave_power*active_width)

                # Update Z values based on your equations
                average_generatable_power_matrix[i, j] = average_generatable_power
                anual_production_matrix[i, j] = anual_production
                capacity_factor_matrix[i, j] = capacity_factor
                overall_eff_matrix[i, j] = overall_eff

                current_iteration += 1
                progress_percentage = current_iteration / total_iterations
                progress_bar.progress(int(progress_percentage * 100))



        def plot_3d_matrix_1(matrix, title):
            fig = go.Figure(data=[go.Surface(z=matrix, x=rotor_size_range, y=install_capacity_range, colorscale='inferno')])
            fig.update_traces(contours_x=dict( usecolormap=True,
                                    highlightcolor="red" ),contours_y=dict(
                                    highlightcolor="limegreen"),contours_z=dict(show=True, usecolormap=True, project_z=True))
            fig.update_layout(
                title=title,
                margin=dict(l=20, r=20, b=20, t=20),
                                        height=450,
                                        autosize=False,
                scene=dict(
                    

                    xaxis_title='Rotor size (m)',
                    yaxis_title='Install capacity (MW)',
                    zaxis_title=title,
                    
                )
            
            )
            st.plotly_chart(fig)

        col2, col3, col4  = st.columns(3)



       # with col1:
        # Display 3D plots with different perspectives
            
        #    plot_3d_matrix_1(average_generatable_power_matrix,"Average power generation")
        with col2:

            plot_3d_matrix_1(anual_production_matrix,"Anual energy production")

        with col3:
            plot_3d_matrix_1(capacity_factor_matrix,"Capacity factor")
        with col4:
            plot_3d_matrix_1(overall_eff_matrix,"Overall Efficiency")





    if st.button("Generate 3D Plot"):
        gen_end(non_dim_curve,Tm_array,Hm0_array,wave_power,prob_matrix)























