# Makefile for POET
#
# `make` - Compile C routines used by POET and compile the User Manual
# LaTeX document
#
all: make_cent make_userman make_mccubed make_models_c make_apphot
clean: clean_userman


clean_userman:
	@echo "\nRemoving user manual..."
	@cd doc/userman && rm -f poet_user_manual.pdf
	@echo "Done."

make_cent:
	@echo "\nBuilding centering package..."
	@cd lib/la && ./build.sh 3
	@echo "Finished building centering package."

make_mccubed:
	@echo "\nBuilding MCcubed package..."
	@cd lib/mccubed && make PY3=1
	@echo "Finished building MCcubed package."

make_models_c:
	@echo "\nBuilding models_c files..."
	@cd lib/models_c && ./build.sh
	@echo "\nFinished building models_c files."

make_userman:
	@echo "\nCompiling user manual..."
	@cd doc/userman && pdflatex poet_user_manual.tex && pdflatex poet_user_manual.tex
	@echo "Finished compiling user manual."

make_apphot:
	@echo "\nBuilding aperture photometry routine..."
	@cd lib && python3 setup_apphot.py install --install-lib='.'
	@echo "Finished building aperture photometry routine."
