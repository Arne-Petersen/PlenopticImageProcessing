/**
 * Copyright 2019 Arne Petersen, Kiel University
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
 *    associated documentation files (the "Software"), to deal in the Software without restriction, including
 *    without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
 *    sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject
 *    to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in all copies or substantial
 *    portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
 *    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
 *    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 *    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <iostream>

#include "PIPBase/DataIO.hh"
#include "PIPAlgorithms/PlenopticTools.hh"
#include "PIPAlgorithms/CUDA/DisparityRefinement_Crosscheck.hh"

using namespace std;
using namespace PIP;

int main(int argc, char** argv)
{
	if (argc != 5)
	{
		cout << "Usage : " << argv[0] << " inputfile inputcalib floatTolerance outputfile" << endl;
		cout << "\t inputfile     \t: single channel single precision floating point images containing normalized disparities" << endl;
		cout << "\t inputcalib    \t: xml file containing MLA configuration" << endl;
		cout << "\t floatTolerance\t: disparity tolerance for filter in pixel (from 0 strict filter to 'microlens diam' only smoothing)" << endl;
		cout << "\t outputfile    \t: exr file to write output to as normalized disparities" << endl;
		return -1;
	}

	string strRawfepthFilename;
	string strMlaCalibFilename;
	string strOutputFilename;
	double fTolerance;

	strRawfepthFilename = argv[1];
	strMlaCalibFilename = argv[2];
	fTolerance = atof(argv[3]);
	strOutputFilename = argv[4];

	cout << "Starting filtering on" << endl;
	cout << "\t Map : " << strRawfepthFilename << endl;
	cout << "\t MLA : " << strMlaCalibFilename << endl;
	cout << "\t tol : " << fTolerance << endl;

	try
	{
		// read raw depth an MLA config
		CVImage_sptr spRawMap(new CVImage());
		CDataIO::ImportImage(*spRawMap, strRawfepthFilename, false);
		if (spRawMap->type() != CV_32FC1)
		{
			throw CRuntimeException("Only single-channel single precision floating point images supported.",
				ERuntimeExcpetionType::ILLEGAL_ARGUMENT);
		}
		SPlenCamDescription descrMLA;
		descrMLA.Reset();
		CPlenopticTools::ReadMlaDescription(descrMLA, strMlaCalibFilename);

		// Set parameters for checker and apply filter
		CCUDADisparityRefinement_Crosscheck cChecker;
		map<string, double> mapParams;
		mapParams["Max Disp Difference"] = fTolerance;
		cChecker.SetParameters(descrMLA, mapParams);
		cChecker.RefineDisparities(spRawMap, spRawMap);

		// write output
		cout << "exporting to " << strOutputFilename << endl;
		CDataIO::ExportImage(*spRawMap, strOutputFilename);
	}
	catch (exception& exc)
	{
		cerr << "Critical error :" << endl;
		cerr << exc.what() << endl;
		return -1;
	}

    return 0;
}


