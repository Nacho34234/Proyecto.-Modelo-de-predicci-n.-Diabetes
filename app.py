from flask import Flask, render_template_string, request, redirect, url_for, flash, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.secret_key = 'supersecretkey'


data = pd.read_csv('C:/Users/NACHO-PC/Desktop/Proyecto/archive/diabetes.csv')
data_cleaned = data.drop(['SkinThickness'], axis=1)


median_insulin = data_cleaned['Insulin'].replace(0, pd.NA).median()
data_cleaned['Insulin'] = data_cleaned['Insulin'].replace(0, median_insulin)


X = data_cleaned.drop(['Outcome', 'DiabetesPedigreeFunction', 'Pregnancies'], axis=1)
y = data_cleaned['Outcome']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)


X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.70, random_state=42)


param_grid_lr = {'C': [0.01, 0.1, 1, 10], 'max_iter': [1000, 2000, 3000]}
grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_


param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'class_weight': ['balanced', 'balanced_subsample', None]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_


best_model = best_rf if accuracy_score(y_val, best_rf.predict(X_val)) > accuracy_score(y_val, best_lr.predict(X_val)) else best_lr


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


def create_tables():
    conn = get_db_connection()

    
    conn.execute('DROP TABLE IF EXISTS predictions')

    
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        keyword TEXT NOT NULL  -- Nueva columna para la palabra clave
    )''')
    
    conn.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        glucose INTEGER,
        blood_pressure INTEGER,
        insulin INTEGER,
        bmi REAL,
        age INTEGER,
        prediction TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    
    conn.commit()
    conn.close()

create_tables()


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        keyword = request.form['keyword']  

        conn = get_db_connection()
        existing_user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()

        if existing_user:
            flash('Este correo electrónico ya ha sido registrado.')
        else:
            conn.execute('INSERT INTO users (first_name, last_name, email, password, keyword) VALUES (?, ?, ?, ?, ?)',
                         (first_name, last_name, email, password, keyword))
            conn.commit()
            flash('Registro exitoso. Ahora puedes iniciar sesión.')
            return redirect(url_for('login'))
        
        conn.close()
    
    return render_template_string('''
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            width: 300px;
        }
        h2 {
            text-align: center;
        }
        input {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            width: 100%;
            padding: 10px;
            background-color: #5cb85c;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #4cae4c;
        }
        a {
            display: block;
            text-align: center;
            margin-top: 10px;
            color: #007bff;
            text-decoration: none;
        }
    </style>
    <div class="container">
        <h2>Registro</h2>
        {% with messages = get_flashed_messages() %} 
          {% if messages %} 
            <ul> 
            {% for message in messages %} 
              <li>{{ message }}</li> 
            {% endfor %} 
            </ul> 
          {% endif %} 
        {% endwith %} 
        <form method="POST"> 
            <input type="text" name="first_name" placeholder="Nombre" required> 
            <input type="text" name="last_name" placeholder="Apellido" required> 
            <input type="email" name="email" placeholder="Email" required> 
            <input type="password" name="password" placeholder="Contraseña" required> 
            <input type="text" name="keyword" placeholder="Palabra Clave" required>  <!-- Campo de palabra clave -->
            <button type="submit">Registrar</button> 
        </form> 
        <a href="/login">Ya tengo cuenta</a>  
    </div>
    ''')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            return redirect(url_for('predict_form'))
        else:
            flash('El correo o la contraseña ingresados no son correctos.')
    
    return render_template_string(''' 
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            width: 300px;
        }
        h2 {
            text-align: center;
        }
        input {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            width: 100%;
            padding: 10px;
            background-color: #5cb85c;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #4cae4c;
        }
        a {
            display: block;
            text-align: center;
            margin-top: 10px;
            color: #007bff;
            text-decoration: none;
        }
    </style>
    <div class="container">
        <h2>Iniciar Sesión</h2>
        {% with messages = get_flashed_messages() %} 
          {% if messages %} 
            <ul> 
            {% for message in messages %} 
              <li>{{ message }}</li> 
            {% endfor %} 
            </ul> 
          {% endif %} 
        {% endwith %} 
        <form method="POST"> 
            <input type="email" name="email" placeholder="Email" required> 
            <input type="password" name="password" placeholder="Contraseña" required> 
            <button type="submit">Iniciar Sesión</button> 
        </form> 
        <a href="/register">Registrarse</a>
        <a href="/forgot_password">Olvidé la contraseña</a>  <!-- Enlace para recuperar la contraseña -->
    </div>
    ''')


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        keyword = request.form['keyword']
        new_password = generate_password_hash(request.form['new_password'])

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND keyword = ?', (email, keyword)).fetchone()
        
        if user:
            conn.execute('UPDATE users SET password = ? WHERE email = ?', (new_password, email))
            conn.commit()
            flash('Contraseña cambiada con éxito.')
            return redirect(url_for('login'))
        else:
            flash('Email o palabra clave incorrectos.')

        conn.close()

    return render_template_string('''
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            width: 300px;
        }
        h2 {
            text-align: center;
        }
        input {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            width: 100%;
            padding: 10px;
            background-color: #5cb85c;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #4cae4c;
        }
    </style>
    <div class="container">
        <h2>Olvidé la Contraseña</h2>
        {% with messages = get_flashed_messages() %} 
          {% if messages %} 
            <ul> 
            {% for message in messages %} 
              <li>{{ message }}</li> 
            {% endfor %} 
            </ul> 
          {% endif %} 
        {% endwith %} 
        <form method="POST"> 
            <input type="email" name="email" placeholder="Email" required> 
            <input type="text" name="keyword" placeholder="Palabra Clave" required> 
            <input type="password" name="new_password" placeholder="Nueva Contraseña" required> 
            <button type="submit">Cambiar Contraseña</button> 
        </form> 
    </div>
    ''')


@app.route('/predict', methods=['GET', 'POST'])
def predict_form():
    if request.method == 'POST':
        glucose = request.form['glucose']
        blood_pressure = request.form['blood_pressure']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        age = request.form['age']

        
        features = [[float(glucose), float(blood_pressure), float(insulin), float(bmi), int(age)]]
        prediction = best_model.predict(features)[0]

        
        conn = get_db_connection()
        conn.execute('INSERT INTO predictions (user_id, glucose, blood_pressure, insulin, bmi, age, prediction) VALUES (?, ?, ?, ?, ?, ?, ?)',
                     (session['user_id'], glucose, blood_pressure, insulin, bmi, age, prediction))
        conn.commit()
        conn.close()

        return redirect(url_for('prediction_results'))

    return render_template_string('''
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            width: 300px;
        }
        h2 {
            text-align: center;
        }
        input {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            width: 100%;
            padding: 10px;
            background-color: #5cb85c;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #4cae4c;
        }
    </style>
    <div class="container">
        <h2>Predicción de Diabetes</h2>
        <form method="POST"> 
            <input type="number" name="glucose" placeholder="Glucosa" required> 
            <input type="number" name="blood_pressure" placeholder="Presión Arterial" required> 
            <input type="number" name="insulin" placeholder="Insulina" required> 
            <input type="number" name="bmi" placeholder="Índice de Masa Corporal" required> 
            <input type="number" name="age" placeholder="Edad" required> 
            <button type="submit">Predecir</button> 
        </form> 
    </div>
    ''')


@app.route('/prediction_results')
def prediction_results():
    conn = get_db_connection()

    
    predictions = conn.execute('SELECT * FROM predictions WHERE user_id = ?', (session['user_id'],)).fetchall()
    
    
    last_prediction = conn.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1', (session['user_id'],)).fetchone()

    conn.close()

    def interpretar_prediccion(prediccion_binaria):
        
        if prediccion_binaria == b'\x01\x00\x00\x00\x00\x00\x00\x00' or prediccion_binaria == 1:
            return "Tiene diabetes"
        elif prediccion_binaria == b'\x00\x00\x00\x00\x00\x00\x00\x00' or prediccion_binaria == 0:
            return "No tiene diabetes"
        else:
            return "Predicción desconocida"

    
    if last_prediction:
        prediccion_interpretada = interpretar_prediccion(last_prediction['prediction'])
    else:
        prediccion_interpretada = None

    return render_template_string('''
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .container {
            width: 80%;
            margin: auto;
            padding: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        table, th, td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
        }
        h2 {
            text-align: center;
        }
        .prediction-result {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            text-align: center;
        }
    </style>

    <div class="container">
        <h2>Resultados de Predicción</h2>
        
        {% if last_prediction %}
        <div class="prediction-result">
            <h3>Tu última predicción</h3>
            <p><strong>Glucosa:</strong> {{ last_prediction['glucose'] }}</p>
            <p><strong>Presión Arterial:</strong> {{ last_prediction['blood_pressure'] }}</p>
            <p><strong>Insulina:</strong> {{ last_prediction['insulin'] }}</p>
            <p><strong>IMC:</strong> {{ last_prediction['bmi'] }}</p>
            <p><strong>Edad:</strong> {{ last_prediction['age'] }}</p>
            <p><strong>Predicción:</strong> {{ prediccion_interpretada }}</p>
            <p><strong>Fecha:</strong> {{ last_prediction['timestamp'] }}</p>
        </div>
        {% else %}
        <p>No se encontraron predicciones.</p>
        {% endif %}

        <table>
            <tr>
                <th>ID</th>
                <th>Glucosa</th>
                <th>Presión Arterial</th>
                <th>Insulina</th>
                <th>IMC</th>
                <th>Edad</th>
                <th>Predicción</th>
                <th>Fecha</th>
            </tr>
            {% for pred in predictions %}
            <tr>
                <td>{{ pred['id'] }}</td>
                <td>{{ pred['glucose'] }}</td>
                <td>{{ pred['blood_pressure'] }}</td>
                <td>{{ pred['insulin'] }}</td>
                <td>{{ pred['bmi'] }}</td>
                <td>{{ pred['age'] }}</td>
                <td>{{ interpretar_prediccion(pred['prediction']) }}</td>
                <td>{{ pred['timestamp'] }}</td>
            </tr>
            {% endfor %}
        </table>

        <a href="/download_pdf">Descargar PDF</a>
        <a href="/logout">Cerrar sesión</a>
    </div>
    ''', last_prediction=last_prediction, predictions=predictions, interpretar_prediccion=interpretar_prediccion)

@app.route('/download_pdf')
def download_pdf():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    predictions = conn.execute('SELECT * FROM predictions WHERE user_id = ?', (session['user_id'],)).fetchall()
    conn.close()

    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    
    width, height = letter 
    margin_left = 50
    margin_top = 750
    line_height = 20  
    page_limit = 40  
    
   
    def draw_header(pdf, y_position):
        pdf.drawString(margin_left, y_position, "ID")
        pdf.drawString(margin_left + 30, y_position, "Glucosa")
        pdf.drawString(margin_left + 90, y_position, "Pres.Art")
        pdf.drawString(margin_left + 150, y_position, "Insulina")
        pdf.drawString(margin_left + 200, y_position, "IMC")
        pdf.drawString(margin_left + 250, y_position, "Edad")
        pdf.drawString(margin_left + 300, y_position, "Predicción")
        pdf.drawString(margin_left + 400, y_position, "Fecha")

    
    def interpretar_prediccion(prediccion_binaria):
        
        if isinstance(prediccion_binaria, (bytes, bytearray)):
            
            prediccion_binaria = int.from_bytes(prediccion_binaria, byteorder='little')
        
        
        if prediccion_binaria == 1:
            return "Tiene diabetes"
        elif prediccion_binaria == 0:
            return "No tiene diabetes"
        else:
            return "Predicción desconocida"

    y = margin_top
    draw_header(pdf, y)
    y -= line_height

    for index, prediction in enumerate(predictions):
        if y < 50:  
            pdf.showPage()  
            y = margin_top  
            draw_header(pdf, y)
            y -= line_height

        pdf.drawString(margin_left, y, str(prediction['id']))
        pdf.drawString(margin_left + 50, y, str(prediction['glucose']))
        pdf.drawString(margin_left + 100, y, str(prediction['blood_pressure']))
        pdf.drawString(margin_left + 150, y, str(prediction['insulin']))
        pdf.drawString(margin_left + 200, y, str(prediction['bmi']))
        pdf.drawString(margin_left + 250, y, str(prediction['age']))

        
        prediccion_interpretada = interpretar_prediccion(prediction['prediction'])
        pdf.drawString(margin_left + 300, y, prediccion_interpretada)

        pdf.drawString(margin_left + 400, y, prediction['timestamp'])
        y -= line_height  

    pdf.save()
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, as_attachment=True, download_name='historial_predicciones.pdf')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Has cerrado sesión.')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
