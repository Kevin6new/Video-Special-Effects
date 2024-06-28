from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app)
cpp_executable_path = r"C:\Users\kevin\source\repos\extension\x64\Release\extension.exe"

def run_cpp_filter_command(filter_name):
    if not os.path.isfile(cpp_executable_path):
        return jsonify({"error": "C++ executable not found"}), 404

    try:
        subprocess.Popen([cpp_executable_path, filter_name])
        return jsonify({"message": f"{filter_name} filter applied"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/apply-greyscale', methods=['GET'])
def apply_greyscale():
    return run_cpp_filter_command("greyscale")

@app.route('/apply-sepia', methods=['GET'])
def apply_sepia():
    return run_cpp_filter_command("sepia")

@app.route('/apply-blur5x5', methods=['GET'])
def apply_blur5x5():
    return run_cpp_filter_command("blur5x5")

@app.route('/apply-blur5x5-alt', methods=['GET'])
def apply_blur5x5_alt():
    return run_cpp_filter_command("blur5x5-alt")

@app.route('/apply-sobel-x', methods=['GET'])
def apply_sobel_x():
    return run_cpp_filter_command("sobel-x")

@app.route('/apply-sobel-y', methods=['GET'])
def apply_sobel_y():
    return run_cpp_filter_command("sobel-y")

@app.route('/apply-magnitude', methods=['GET'])
def apply_magnitude():
    return run_cpp_filter_command("magnitude")

@app.route('/apply-blur-quantize', methods=['GET'])
def apply_blur_quantize():
    return run_cpp_filter_command("blur-quantize")

# Additional routes for new functionalities
@app.route('/apply-detect-faces', methods=['GET'])
def apply_detect_faces():
    return run_cpp_filter_command("detect-faces")

@app.route('/apply-cartoon', methods=['GET'])
def apply_cartoon():
    return run_cpp_filter_command("cartoon")

@app.route('/apply-negative', methods=['GET'])
def apply_negative():
    return run_cpp_filter_command("negative")

@app.route('/apply-colorize-faces', methods=['GET'])
def apply_colorize_faces():
    return run_cpp_filter_command("colorize-faces")

@app.route('/apply-increase-brightness', methods=['GET'])
def apply_increase_brightness():
    return run_cpp_filter_command("increase-brightness")

@app.route('/apply-decrease-brightness', methods=['GET'])
def apply_decrease_brightness():
    return run_cpp_filter_command("decrease-brightness")

@app.route('/apply-increase-contrast', methods=['GET'])
def apply_increase_contrast():
    return run_cpp_filter_command("increase-contrast")

@app.route('/apply-decrease-contrast', methods=['GET'])
def apply_decrease_contrast():
    return run_cpp_filter_command("decrease-contrast")

@app.route('/quit', methods=['GET'])
def quit():
    return jsonify({"message": "Quit command sent"})

if __name__ == '__main__':
    app.run(port=8000)
