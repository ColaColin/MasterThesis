all: heft

heft:
	wiki2beamer content.txt > content.tex
	rubber -d presentation.tex
	

clean:
	rm *.aux *.log *.nav *.out *.pdf *.snm content.tex *.toc *.vrb -f
