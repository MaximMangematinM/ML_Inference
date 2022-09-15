
from flask import Flask
from commons import preprocess_image, get_prediction, image_classification_vit
from flask import render_template, redirect, request
import os
import json
from sentiment_annalisis import eval_sentence_pipeline_roberta_twitter, eval_sentence_roberta_large

app = Flask(__name__)


@app.route("/image_classification_result")
def densenet_display_result():
    """Function that calls the image classification model and render the result into the app"""
    with open("classification_result.json", "r") as f:
        classification_result = json.load(f) #retrieve the result from the json
    return render_template('predict.html', class_id=classification_result["class_id"],
                               class_name=classification_result["class_name"])
    


@app.route('/import_image', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file') #get the file sored into the request
        if not file:
            raise FileNotFoundError("No file found")
        file.save(os.path.join(app.root_path, 'static/image.png')) #save the file into the static folder

        #model selection
        try:
            request.form["vit"]
            class_name, class_id = image_classification_vit("./static/image.png")
        except KeyError:
            try:
                request.form["den"]
                input = preprocess_image("./static/image.png") #preprocess the image (crop, rezise..)
                class_name, class_id = get_prediction(input)
            except KeyError:
                return render_template("index.html")
        
        #save the result into an json
        with open("classification_result.json", "w") as res_file:
            json.dump({"class_id" : class_id, "class_name" : class_name}, res_file)
        
        return redirect("/image_classification_result") #go to the result page


    return render_template("index.html")


@app.route("/", methods=["POST", "GET"])
def choose_model():
    if request.method == "POST":
        values = request.form
        print(values)

        #model selection
        try:
            values["img"]
            return redirect("/import_image")
        except KeyError:
            try:
                values["text"]
                return redirect("/get_sentence")
            except KeyError:
                return render_template("choose.html")    
    return render_template("choose.html")


@app.route("/get_sentence", methods=["POST", "GET"])
def enter_sentence():

    if request.method == "POST":
        text = request.form["class"] #retrieve the rext

        #model selection
        try :
            request.form["small"]
            text_class = eval_sentence_pipeline_roberta_twitter(text)
        except KeyError:
            try:
                request.form["large"]
                text_class = eval_sentence_roberta_large(text)
            except KeyError:
                return render_template("enter_text.html")
        
        with open("./result/text_class_result.json", "w") as res_json:
            json.dump({"text" : text, "class" : text_class}, res_json)
        
        return redirect("/sentence_sentiment")
    return render_template("enter_text.html")



@app.route("/sentence_sentiment")
def text_classification_restult():
    with open("/result/text_class_result.json", "r") as f:
        res = json.load(f)
    sentiment = res["class"]
    sentence = res["text"]

    return render_template("res_text_classification.html", sentence = sentence, sentiment = sentiment)



if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))