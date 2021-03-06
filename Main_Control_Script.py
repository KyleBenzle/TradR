import subprocess
from github import Github
import pandas as pd
import glob
import os
import csv
from time import sleep
import datetime


count = 0

while count < 7:
	count = count + 1
	exec(open("RedditScraper.py").read())



	sleep(3)




	list_of_files = glob.glob('./*.csv') # * means all if need specific format then *.csv
	latest_file = max(list_of_files, key=os.path.getctime)
	latestFile = pd.read_csv(latest_file, encoding ='utf-8')
	latestFile = latestFile.iloc[0:1]

	latestFile.Hour=pd.datetime.now()


	mainFile = pd.read_csv('./ScrappedData/ScrappedReddit.csv', encoding='utf-8')

	# latestFile.to_csv('lattestfiletest.csv')


	os.remove("./ScrappedData/ScrappedReddit.csv")

	os.remove(latest_file)

	# os.remove(latest_file)

	sleep(3)

	latestFileNew = latestFile[mainFile.columns]

	out = pd.concat([mainFile, latestFileNew])

	# out = mainFile.append(latestFile, ignore_index=True)


	out.to_csv('./ScrappedData/ScrappedReddit.csv', index=False)



	# 

	sleep(2)

	exec(open("PriceApp.py").read())

	sleep(2)

	exec(open("Analysis.py").read())

	sleep(2)



	##### Upload to GitHub
	# Set user info
	g = Github("c04f0be78e2e12d4bf6ece07d417ea759c7bd7f5")
	repo = g.get_user().get_repo('tradr')
	all_files = []
	contents = repo.get_contents("")
	while contents:
	    file_content = contents.pop(0)
	    if file_content.type == "dir":
	        contents.extend(repo.get_contents(file_content.path))
	    else:
	        file = file_content
	        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


	# Price
	with open('./PriceData.csv', 'r') as file:
	    content = file.read()

	# Upload Scrapped Reddit Data
	git_prefix = 'Data/'
	git_file = git_prefix + 'PriceData.csv'

	if git_file in all_files:
	    contents = repo.get_contents(git_file)
	    repo.update_file(contents.path, "committing files", content, contents.sha, branch="main")
	    print(git_file + ' UPDATED')
	else:
	    repo.create_file(git_file, "committing files", content, branch="main")
	    print(git_file + ' CREATED')

	    
	# Scrapped Reddit
	with open('./ScrappedData/ScrappedReddit.csv', 'r') as file:
	    content = file.read()

	# Upload Scrapped Reddit Data
	git_prefix = 'Data/'
	git_file = git_prefix + 'ScrappedReddit.csv'

	if git_file in all_files:
	    contents = repo.get_contents(git_file)
	    repo.update_file(contents.path, "committing files", content, contents.sha, branch="main")
	    print(git_file + ' UPDATED')
	else:
	    repo.create_file(git_file, "committing files", content, branch="main")
	    print(git_file + ' CREATED')



	# Signal
	with open('./SignalInput.csv', 'r') as file:
	    content = file.read()

	# Upload Scrapped Reddit Data
	git_prefix = 'Data/'
	git_file = git_prefix + 'SignalInput.csv'

	if git_file in all_files:
	    contents = repo.get_contents(git_file)
	    repo.update_file(contents.path, "committing files", content, contents.sha, branch="main")
	    print(git_file + ' UPDATED')
	else:
	    repo.create_file(git_file, "committing files", content, branch="main")
	    print(git_file + ' CREATED')

# user input for API KEY




	sleep(2)

	# Make trades
	main_path = '/home/i/MEGA/NYCData/tradr/Data/TradeMaker/'
	python_path = f"{main_path}venv/bin/python3"
	args = [python_path, f"{main_path}trade_script.py",
	"/home/i/MEGA/NYCData/tradr/Data/SignalInput.csv"]
	process_info = subprocess.run(args)
	print(process_info.returncode)

	print('waiting...')
	sleep(3600)




	# analysis Code



# run the trade



# fix colum names so they are the same!!!!




# for each in latest_file:
# 	if each[0]=='Hour':
# 		continue
# 	mainfile.append(each)


# with open("./main/RedditScrapped1.csv",'a+',encoding='utf-16') as outfile:
# 	writer = csv.writer(outfile)
# 	for each in mainfile:
# 		writer.writerow(each)
