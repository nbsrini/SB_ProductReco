# This file provides the app interface - this imports the model_interface to get the recommendations
## User: Nitin Balaji Srinivasan AI& ML (Cohort 58)
import nltk 
print("NLTK Version:", nltk.__version__)
# Download NLTK corpora before starting the app
nltk.download('punkt')  # Replace with the corpora you need, e.g., 'stopwords', 'wordnet'
nltk.download('punkt_tab') # This line is added to download the missing resource.

from flask import Flask,render_template,request
import model_interface
app = Flask('__name__')

selection_reviewusernames = ['zzz1127','piggyboy420','zburt5','joshua','dorothy','cassie','moore222','rebecca','walker557','samantha','raeanne','kimmie']
@app.route('/')
def view():
    return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend_top5():
    print(request.method)
    user_name = request.form['User Name']
    print('User name=',user_name)
    
    if  user_name in selection_reviewusernames and request.method == 'POST':
            top20_products = model_interface.suggest_products(user_name)
            print(top20_products.head())
            get_top5 = model_interface.top_rated_products(top20_products)
            
            return render_template('index.html',column_names=get_top5.columns.values, row_data=list(get_top5.values.tolist()), zip=zip,text='Recommended products')
    elif not user_name in  selection_reviewusernames:
        return render_template('index.html',text='No Recommendations available for the user')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.debug=False

    app.run()