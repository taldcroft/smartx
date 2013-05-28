index:
	python make_index.py

www-reports:
	rsync -av --include '*/' --include='**.png' --include='**.html' --exclude='*' \
	    reports/ ~/www.new/tmp/reports/
