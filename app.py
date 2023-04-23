from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

clf = pickle.load(open('Sclf.pkl',"rb"))
loaded_vec = pickle.load(open("Scount_vect.pkl","rb"))

@app.route('/')
def sentiments():
    return render_template('test.html')


@app.route('/result',methods = ['POST' , 'GET'])
def result():
    if request.method == 'POST':
        result1 = request.form.get('Data1')
        result_pred1 = clf.predict(loaded_vec.transform([result1]))

        result2 = request.form.get('Data2')
        result_pred2 = clf.predict(loaded_vec.transform([result2]))

        result3 = request.form.get('Data3')
        result_pred3 = clf.predict(loaded_vec.transform([result3]))

        result4 = request.form.get('Data4')
        result_pred4 = clf.predict(loaded_vec.transform([result4]))

        result5 = request.form.get('Data5')
        result_pred5 = clf.predict(loaded_vec.transform([result5]))

        a = str(result_pred1)
        b = str(result_pred2)
        c = str(result_pred3)
        d = str(result_pred4)
        e = str(result_pred5)

        from collections import Counter 
        # input_dict = {'A': a}
        input_dict = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}

        value, count = Counter(input_dict.values()).most_common(1)[0]
    

        return render_template("sentiments_result.html", result = value)
    

if __name__ == '__main__':
    app.debug=True
    app.run()    