import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model
with open('vastu_model.pkl', 'rb') as file:
    model = pickle.load(file)



@app.route('/')
def index():
    return render_template('form.html', prediction_result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        entrance_direction = request.form['entrance_direction']
        kitchen_position = request.form['kitchen_position']
        bedroom_position = request.form['bedroom_position']
        plot_shape = request.form['plot_shape']
        floor_number = int(request.form['floor_number'])
        open_space_ne = request.form['open_space_ne']
        water_source_ne = request.form['water_source_ne']
        heavy_objects_sw = request.form['heavy_objects_sw']
        house_area = int(request.form['house_area'])
        road_facing = request.form['road_facing']

        # Prepare the input data
        input_data = {
            'Direction of Main Entrance': [entrance_direction],
            'Position of Kitchen': [kitchen_position],
            'Position of Bedroom': [bedroom_position],
            'Plot Shape': [plot_shape],
            'Floor Number': [floor_number],
            'Is There an Open Space in the North-East?': [open_space_ne],
            'Water Source in North-East': [water_source_ne],
            'Presence of Heavy Objects in South-West': [heavy_objects_sw],
            'House Area (sq ft)': [house_area],
            'Road Facing': [road_facing]
        }

        # Convert the input data to a DataFrame
        input_df = pd.DataFrame(input_data)

        # Manual encode the colums that are label encoded
        input_df['Is There an Open Space in the North-East?'] = [1 if val == 'Yes' else 0 for val in input_df['Is There an Open Space in the North-East?']]
        input_df['Water Source in North-East'] = [1 if val == 'Yes' else 0 for val in input_df['Water Source in North-East']]
        input_df['Presence of Heavy Objects in South-West'] = [1 if val == 'Yes' else 0 for val in input_df['Presence of Heavy Objects in South-West']]

        # Apply One-Hot Encoding to other categorical columns
        input_df = pd.get_dummies(input_df, columns=['Direction of Main Entrance', 'Position of Kitchen', 'Position of Bedroom', 'Plot Shape', 'Road Facing'], drop_first=True)

        # Ensure the columns in input_df match the model’s expected input columns
        model_columns = pickle.load(open('model_columns.pkl', 'rb'))
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Make the prediction
        prediction = model.predict(input_df)
        prediction_result = "Good" if prediction[0] == 1 else "Bad"

        # Render the result back to the HTML page
        return render_template('form.html', prediction_result=prediction_result)

    except Exception as e:
        return render_template('form.html', prediction_result=f'An error occurred: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
