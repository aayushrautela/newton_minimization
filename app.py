from flask import Flask, render_template, request
import os
from lab1_v4 import newton_method, visualize_path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        x = float(request.form['x'])
        y = float(request.form['y'])
        alpha = float(request.form['alpha'])
        
        # Run Newton's method
        (x_min, y_min), iterations, path = newton_method([x, y], alpha)
        plot_name = f"plot_{x}_{y}_{alpha}_iterations_{len(path)}.png"
        visualize_path(path, label=os.path.join(app.config['UPLOAD_FOLDER'], plot_name[:-4]))
        plot_url = plot_name
        
        return render_template('result.html', 
                            x=x, y=y, alpha=alpha,
                            x_min=x_min, y_min=y_min,
                            iterations=iterations,
                            plot_url=plot_name)
    
    return render_template('form.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=52999)