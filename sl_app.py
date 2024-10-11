import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

with open('star_model.pkl','rb') as file:
    model=pickle.load(file)

data = pd.read_csv("data\stars.csv") 
categorical_columns = ['Color', 'Spectral_Class']
data=data.drop(columns=['Type'])

def main():
    st.markdown("<h1 style='color: #8B0000;'>Star Type Prediction App</h1>", unsafe_allow_html=True)

    st.sidebar.header("User Input")

    user_input = {}
    for column in data.columns:
        if column != 'Type':
            if column in categorical_columns:
                unique_values = data[column].unique()
                selected_value = st.sidebar.selectbox(f"{column}:", unique_values)
                user_input[column] = selected_value
            else:
                user_input[column] = st.sidebar.text_input(f"{column}:", float(data[column].mean()))
    user_input_df = pd.DataFrame(user_input, index=[0])
    data_with_user_input = pd.concat([data, user_input_df], ignore_index=True)
    data_with_user_input = pd.get_dummies(data_with_user_input, columns=categorical_columns)
    #print(data_with_user_input.columns)
    numerical_columns = [col for col in data_with_user_input.columns if col not in categorical_columns]
    scaler = StandardScaler()
    data_with_user_input[numerical_columns] = scaler.fit_transform(data_with_user_input[numerical_columns])
   
    user_input_df = data_with_user_input.iloc[-1, :]
    data_with_user_input = data_with_user_input.iloc[:-1, :]
    #print(user_input_df)
    prediction = model.predict(user_input_df.values.reshape(1, -1))
    print(user_input_df.values.reshape(1, -1),prediction)
    category_mapping = {
    0: f'''**{"Red Dwarf"}** :  
      
    A red dwarf is the smallest and coolest kind of star on the main sequence. 
    Red dwarfs are by far the most common type of star in the Milky Way, at least in the neighborhood of the Sun. 
    However, individual red dwarfs cannot be easily observed as a result of their low luminosity. 
    From Earth, not one star that fits the stricter definitions of a red dwarf is visible to the naked eye.
    Proxima Centauri, the nearest star to the Sun, is a red dwarf, as are fifty of the sixty nearest stars. 
    According to some estimates, red dwarfs make up three-quarters of the stars in the Milky Way.''',
    1: f'''**{"Brown Dwarf"}** :  
      
    Brown dwarfs are substellar objects that have more mass than the biggest gas giant planets, 
    but less than the least massive main-sequence stars. 
    Their mass is approximately 13 to 80 times that of Jupiter (MJ)—not big enough to sustain nuclear fusion of ordinary hydrogen (1H) 
    into helium in their cores, but massive enough to emit some light and heat from the fusion of deuterium (2H). 
    The most massive ones (> 65 MJ) can fuse lithium (7Li).''',
    2: f'''**{"White Dwarf"}** :  
      
    A white dwarf is a stellar core remnant composed mostly of electron-degenerate matter. 
    A white dwarf is very dense: its mass is comparable to the Sun's, while its volume is comparable to Earth's. 
    A white dwarf's low luminosity comes from the emission of residual thermal energy; no fusion takes place in a white dwarf.
    The nearest known white dwarf is Sirius B, at 8.6 light years, the smaller component of the Sirius binary star. 
    There are currently thought to be eight white dwarfs among the hundred star systems nearest the Sun.''',
    3: f'''**{"Main Sequence"}** :  
      
    In astronomy, the main sequence is a classification of stars which appear on plots of stellar color versus brightness 
    as a continuous and distinctive band. Stars on this band are known as main-sequence stars or dwarf stars, 
    and positions of stars on and off the band are believed to indicate their physical properties, 
    as well as their progress through several types of star life-cycles. 
    These are the most numerous true stars in the universe and include the Sun. ''',
    4: f'''**{"Supergiant"}** :  
      
    Supergiants are among the most massive and most luminous stars. 
    Supergiant stars occupy the top region of the Hertzsprung–Russell diagram with absolute visual magnitudes between about −3 and −8.
    the temperature range of supergiant stars spans from about 3,400 K to over 20,000 K.''',
    5: f'''**{"Hypergiant"}** :  
      
    A hypergiant (luminosity class 0 or Ia+) is a very rare type of star that has an extremely high luminosity, mass, size 
    and mass loss because of its extreme stellar winds. The term hypergiant is defined as luminosity class 0 (zero) in the MKK system. 
    However, this is rarely seen in literature or in published spectral classifications, except for specific well-defined groups 
    such as the yellow hypergiants, RSG (red supergiants), or blue B(e) supergiants with emission spectra.'''
    }

    predicted_category = category_mapping[prediction[0]]
   
    st.subheader("The predicted star type is:")
    st.write(f" {predicted_category}")

if __name__ == '__main__':
    main()
