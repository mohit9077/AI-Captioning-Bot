from flask import Flask, render_template, redirect, request

import final

app=Flask(__name__)



@app.route('/')  ## routes are used for handling the different urls
def hello():
	return render_template("index.html")
	# we will write all the html code in this file
	#flask already know that html file will be present in templates folder

@app.route('/',methods=['POST'])
def submit():
	if request.method=='POST':

		f=request.files['userfile']
		path="./static/{}".format(f.filename)
		f.save(path)

		caption=final.caption_this_image(path)
		
		result_dic={
		'image': path,
		'caption':caption
		}

	return render_template("index.html", result = result_dic)

if __name__=='__main__':
	#app.debug= True ## if you don not set this to true then to see every small changes ,you need to restart the server
	app.run(debug=True)