
import nibabel as nib
import pandas as pd
import os
import glob
from os.path import join as opj
import pydicom
import json

###1###
# Function to extract parameters from NIfTI file
def extract_parameters(filename):
    img = nib.load(filename)
    header = img.header

    # Extracting parameters
    te = header.get('descrip', b'').tobytes().decode('utf-8')
    if 'TE=' in te:
        te = float(te.split('TE=')[1].split(';')[0])
    else:
        te = None
    
    tr = header['pixdim'][4]
    flip_angle = None  # Flip angle is not available in the provided header, usually found in DICOM
    matrix = header['dim'][1:4]
    number_of_slices = header['dim'][3]
    slice_thickness = header['pixdim'][3]
    voxel_size = header['pixdim'][1:4]

    return [os.path.basename(filename), te, tr, flip_angle, matrix, number_of_slices, slice_thickness, voxel_size]

# List of NIfTI files
MRI_Path = '/../..'
nifti_files = glob.glob(MRI_Path + "/*.nii.gz")

# Create a DataFrame to store the parameters
columns = ['Filename', 'TE (ms)', 'TR (ms)', 'Flip Angle (degrees)', 'Matrix', 'Number of Slices', 'Slice Thickness (mm)', 'Voxel Size (mm)']
data = []
for file in nifti_files:
    data.append(extract_parameters(file))

df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv(opj(MRI_Path + '/mri_parameters.csv'), index=False)
print("Parameters have been saved to mri_parameters.csv")

####2####
# Function to extract parameters from DICOM file
def extract_parameters_dicom(filename):
    dicom_data = pydicom.dcmread(filename)
    # Extracting parameters
    te = None
    flip_angle = None
    
    if 'EchoTime' in dicom_data:
        te = dicom_data.EchoTime
        
    if 'FlipAngle' in dicom_data:
        flip_angle = dicom_data.FlipAngle

    if 'Manufacturer' in dicom_data:
        scanner_manufacturer = dicom_data.Manufacturer

    if 'ManufacturerModelName' in dicom_data:
        scanner_model = dicom_data.ManufacturerModelName

    if 'ScanningSequence' in dicom_data:
        scanning_sequence = dicom_data.ScanningSequence

    if 'SequenceName' in dicom_data:
        sequence_name = dicom_data.SequenceName
    
    if 'SeriesDescription' in dicom_data:
        series_description = dicom_data.SeriesDescription

    if 'ProtocolName' in dicom_data:
        protocal_name = dicom_data.ProtocolName

    return [os.path.split(os.path.split(os.path.dirname(filename))[0])[1], te, flip_angle,
            scanner_manufacturer, 
            scanner_model,
            scanning_sequence, 
            sequence_name,
            series_description,
            protocal_name
            ]

# List of DICOM files
MRI_Path = '/../..'
dicom_files = glob.glob(MRI_Path + "*/dicom/0000000A")

# Create a DataFrame to store the parameters
columns = ['Filename', 'TE (ms)', 'Flip Angle (degrees)',
           'Scanner Manufacturer', 'Scanner Model', 'ScanningSequence', 'SequenceName',
           'SeriesDescription','ProtocolName',
           ]
data = []

for file in dicom_files:
    data.append(extract_parameters_dicom(file))

df = pd.DataFrame(data, columns=columns)

# Save to CSV
out_path = '/../..'
df.to_csv(opj(out_path + '/resend_dicom_parameters.csv'), index=False)
print("Parameters have been saved to dicom_parameters.csv")


###3####
# Load JSON data
MRI_Path = '/../..'
nifti_files = glob.glob(MRI_Path + "/*/*.nii")

# Extract parameters
def extract_parameters(file):

    with open(file, 'r') as f:
        json_data = json.load(f)
        return [
            os.path.split(os.path.dirname(file))[1],
            json_data.get('Manufacturer'),
            json_data.get('ManufacturersModelName'),
            json_data.get('MRAcquisitionType'),
            json_data.get('SeriesDescription'),
            json_data.get('ProtocolName'),
            json_data.get('ScanningSequence'),
            json_data.get('SequenceName'),
            json_data.get('EchoTime'),
            json_data.get('RepetitionTime'),
            json_data.get('FlipAngle')

        ]

parameters = []
Json_Path = '/../..'
for file in nifti_files:
    last_filename = os.path.splitext(os.path.split(file)[1])[0]
    last_directory = os.path.basename(os.path.dirname(file))
    file = os.path.join(Json_Path + last_directory + '/' + last_filename + '.json')
    parameters.append(extract_parameters(file))

# Create a DataFrame
columns = ['Filename', 'Manufacturer', 'ManufacturersModelName', 'MRAcquisitionType', 'SeriesDescription', 'ProtocolName', 'ScanningSequence', 'SequenceName', 'EchoTime', 'RepetitionTime', 'FlipAngle']

df = pd.DataFrame(parameters, columns=columns)

# Save to CSV
out_path = '/../..'
df.to_csv(opj(out_path + '/fsend_dicom_parameters.csv'), index=False)
print("Parameters have been saved to dicom_parameters.csv")


