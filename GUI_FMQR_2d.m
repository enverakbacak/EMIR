function varargout = GUI_FMQR_2d(varargin)
% GUI_FMQR_2D MATLAB code for GUI_FMQR_2d.fig
%      GUI_FMQR_2D, by itself, creates a new GUI_FMQR_2D or raises the existing
%      singleton*.
%
%      H = GUI_FMQR_2D returns the handle to a new GUI_FMQR_2D or the handle to
%      the existing singleton*.
%
%      GUI_FMQR_2D('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI_FMQR_2D.M with the given input arguments.
%
%      GUI_FMQR_2D('Property','Value',...) creates a new GUI_FMQR_2D or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUI_FMQR_2d_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUI_FMQR_2d_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI_FMQR_2d

% Last Modified by GUIDE v2.5 04-Jun-2020 15:06:31

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_FMQR_2d_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_FMQR_2d_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUI_FMQR_2d is made visible.
function GUI_FMQR_2d_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUI_FMQR_2d (see VARARGIN)

% Choose default command line output for GUI_FMQR_2d
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GUI_FMQR_2d wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUI_FMQR_2d_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in Dataset.
function Dataset_Callback(hObject, eventdata, handles)
% hObject    handle to Dataset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Dataset contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Dataset


% --- Executes during object creation, after setting all properties.
function Dataset_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Dataset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on selection change in QueryName1.
function QueryName1_Callback(hObject, eventdata, handles)
% hObject    handle to QueryName1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns QueryName1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from QueryName1

imindex = get(hObject,'Value');
image_dir = handles.image_dir;
fname = [image_dir handles.filenames{imindex}];
axes(handles.axes1);
imshow(imread(fname)); axis image;
handles.q1Idx = imindex;
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function QueryName1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to QueryName1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in QueryName2.
function QueryName2_Callback(hObject, eventdata, handles)
% hObject    handle to QueryName2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns QueryName2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from QueryName2
imindex = get(hObject,'Value');
image_dir = handles.image_dir;
fname = [image_dir handles.filenames{imindex}];
axes(handles.axes2);
imshow(imread(fname)); axis image;
handles.q2Idx = imindex;
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function QueryName2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to QueryName2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes during object creation, after setting all properties.
function FrontSelector_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FrontSelector (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function ImageSelector_Callback(hObject, eventdata, handles)
% hObject    handle to ImageSelector (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider






% --- Executes during object creation, after setting all properties.
function ImageSelector_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ImageSelector (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function FrontNum_Callback(hObject, eventdata, handles)
% hObject    handle to FrontNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of FrontNum as text
%        str2double(get(hObject,'String')) returns contents of FrontNum as a double


% --- Executes during object creation, after setting all properties.
function FrontNum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FrontNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ImageNum_Callback(hObject, eventdata, handles)
% hObject    handle to ImageNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ImageNum as text
%        str2double(get(hObject,'String')) returns contents of ImageNum as a double


% --- Executes during object creation, after setting all properties.
function ImageNum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ImageNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




function main_Callback(hObject, eventdata, handles)
% hObject    handle to main (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of main as text
%        str2double(get(hObject,'String')) returns contents of main as a double


% --- Executes during object creation, after setting all properties.
function main_CreateFcn(hObject, eventdata, handles)
% hObject    handle to main (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function mainR1_Callback(hObject, eventdata, handles)
% hObject    handle to mainR1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of mainR1 as text
%        str2double(get(hObject,'String')) returns contents of mainR1 as a double


% --- Executes during object creation, after setting all properties.
function mainR1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mainR1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function mainR2_Callback(hObject, eventdata, handles)
% hObject    handle to mainR2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of mainR2 as text
%        str2double(get(hObject,'String')) returns contents of mainR2 as a double


% --- Executes during object creation, after setting all properties.
function mainR2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mainR2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function mainR3_Callback(hObject, eventdata, handles)
% hObject    handle to mainR3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of mainR3 as text
%        str2double(get(hObject,'String')) returns contents of mainR3 as a double


% --- Executes during object creation, after setting all properties.
function mainR3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mainR3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function mainL1_Callback(hObject, eventdata, handles)
% hObject    handle to mainL1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of mainL1 as text
%        str2double(get(hObject,'String')) returns contents of mainL1 as a double


% --- Executes during object creation, after setting all properties.
function mainL1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mainL1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function mainL2_Callback(hObject, eventdata, handles)
% hObject    handle to mainL2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of mainL2 as text
%        str2double(get(hObject,'String')) returns contents of mainL2 as a double


% --- Executes during object creation, after setting all properties.
function mainL2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mainL2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function mainL3_Callback(hObject, eventdata, handles)
% hObject    handle to mainL3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of mainL3 as text
%        str2double(get(hObject,'String')) returns contents of mainL3 as a double


% --- Executes during object creation, after setting all properties.
function mainL3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mainL3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function nDCG_Value_Callback(hObject, eventdata, handles)
% hObject    handle to nDCG_Value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of nDCG_Value as text
%        str2double(get(hObject,'String')) returns contents of nDCG_Value as a double


% --- Executes during object creation, after setting all properties.
function nDCG_Value_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nDCG_Value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




function edit13_Callback(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit13 as text
%        str2double(get(hObject,'String')) returns contents of edit13 as a double


% --- Executes during object creation, after setting all properties.
function edit13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit14_Callback(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit14 as text
%        str2double(get(hObject,'String')) returns contents of edit14 as a double


% --- Executes during object creation, after setting all properties.
function edit14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit16_Callback(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit16 as text
%        str2double(get(hObject,'String')) returns contents of edit16 as a double


% --- Executes during object creation, after setting all properties.
function edit16_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes during object creation, after setting all properties.
function tictoc2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tictoc2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




function edit19_Callback(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit19 as text
%        str2double(get(hObject,'String')) returns contents of edit19 as a double


% --- Executes during object creation, after setting all properties.
function edit19_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit20_Callback(hObject, eventdata, handles)
% hObject    handle to edit20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit20 as text
%        str2double(get(hObject,'String')) returns contents of edit20 as a double


% --- Executes during object creation, after setting all properties.
function edit20_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit21_Callback(hObject, eventdata, handles)
% hObject    handle to edit21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit21 as text
%        str2double(get(hObject,'String')) returns contents of edit21 as a double


% --- Executes during object creation, after setting all properties.
function edit21_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit22_Callback(hObject, eventdata, handles)
% hObject    handle to edit22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit22 as text
%        str2double(get(hObject,'String')) returns contents of edit22 as a double


% --- Executes during object creation, after setting all properties.
function edit22_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit23_Callback(hObject, eventdata, handles)
% hObject    handle to edit23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit23 as text
%        str2double(get(hObject,'String')) returns contents of edit23 as a double


% --- Executes during object creation, after setting all properties.
function edit23_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit24_Callback(hObject, eventdata, handles)
% hObject    handle to edit24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit24 as text
%        str2double(get(hObject,'String')) returns contents of edit24 as a double


% --- Executes during object creation, after setting all properties.
function edit24_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit25_Callback(hObject, eventdata, handles)
% hObject    handle to edit25 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit25 as text
%        str2double(get(hObject,'String')) returns contents of edit25 as a double


% --- Executes during object creation, after setting all properties.
function edit25_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit25 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit26_Callback(hObject, eventdata, handles)
% hObject    handle to edit26 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit26 as text
%        str2double(get(hObject,'String')) returns contents of edit26 as a double


% --- Executes during object creation, after setting all properties.
function edit26_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit26 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit27_Callback(hObject, eventdata, handles)
% hObject    handle to edit27 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit27 as text
%        str2double(get(hObject,'String')) returns contents of edit27 as a double


% --- Executes during object creation, after setting all properties.
function edit27_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit27 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit28_Callback(hObject, eventdata, handles)
% hObject    handle to edit28 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit28 as text
%        str2double(get(hObject,'String')) returns contents of edit28 as a double


% --- Executes during object creation, after setting all properties.
function edit28_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit28 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit29_Callback(hObject, eventdata, handles)
% hObject    handle to edit29 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit29 as text
%        str2double(get(hObject,'String')) returns contents of edit29 as a double


% --- Executes during object creation, after setting all properties.
function edit29_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit29 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit30_Callback(hObject, eventdata, handles)
% hObject    handle to edit30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit30 as text
%        str2double(get(hObject,'String')) returns contents of edit30 as a double


% --- Executes during object creation, after setting all properties.
function edit30_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit31_Callback(hObject, eventdata, handles)
% hObject    handle to edit31 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit31 as text
%        str2double(get(hObject,'String')) returns contents of edit31 as a double


% --- Executes during object creation, after setting all properties.
function edit31_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit31 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit32_Callback(hObject, eventdata, handles)
% hObject    handle to edit32 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit32 as text
%        str2double(get(hObject,'String')) returns contents of edit32 as a double


% --- Executes during object creation, after setting all properties.
function edit32_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit32 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit33_Callback(hObject, eventdata, handles)
% hObject    handle to edit33 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit33 as text
%        str2double(get(hObject,'String')) returns contents of edit33 as a double


% --- Executes during object creation, after setting all properties.
function edit33_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit33 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit34_Callback(hObject, eventdata, handles)
% hObject    handle to edit34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit34 as text
%        str2double(get(hObject,'String')) returns contents of edit34 as a double


% --- Executes during object creation, after setting all properties.
function edit34_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit35_Callback(hObject, eventdata, handles)
% hObject    handle to edit35 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit35 as text
%        str2double(get(hObject,'String')) returns contents of edit35 as a double


% --- Executes during object creation, after setting all properties.
function edit35_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit35 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit36_Callback(hObject, eventdata, handles)
% hObject    handle to edit36 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit36 as text
%        str2double(get(hObject,'String')) returns contents of edit36 as a double


% --- Executes during object creation, after setting all properties.
function edit36_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit36 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit37_Callback(hObject, eventdata, handles)
% hObject    handle to edit37 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit37 as text
%        str2double(get(hObject,'String')) returns contents of edit37 as a double


% --- Executes during object creation, after setting all properties.
function edit37_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit37 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit38_Callback(hObject, eventdata, handles)
% hObject    handle to edit38 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit38 as text
%        str2double(get(hObject,'String')) returns contents of edit38 as a double


% --- Executes during object creation, after setting all properties.
function edit38_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit38 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit39_Callback(hObject, eventdata, handles)
% hObject    handle to edit39 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit39 as text
%        str2double(get(hObject,'String')) returns contents of edit39 as a double


% --- Executes during object creation, after setting all properties.
function edit39_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit39 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit40_Callback(hObject, eventdata, handles)
% hObject    handle to edit40 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit40 as text
%        str2double(get(hObject,'String')) returns contents of edit40 as a double


% --- Executes during object creation, after setting all properties.
function edit40_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit40 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit41_Callback(hObject, eventdata, handles)
% hObject    handle to edit41 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit41 as text
%        str2double(get(hObject,'String')) returns contents of edit41 as a double


% --- Executes during object creation, after setting all properties.
function edit41_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit41 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit42_Callback(hObject, eventdata, handles)
% hObject    handle to edit42 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit42 as text
%        str2double(get(hObject,'String')) returns contents of edit42 as a double


% --- Executes during object creation, after setting all properties.
function edit42_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit42 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


   
% --- Executes on selection change in hashCodeSelection_f.
function hashCodeSelection_f_Callback(hObject, eventdata, handles)
% hObject    handle to hashCodeSelection_f (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns hashCodeSelection_f contents as cell array
%        contents{get(hObject,'Value')} returns selected item from hashCodeSelection_f

dataset_index = get(handles.Dataset, 'Value');
switch dataset_index 
    case 1
        image_dir =['Lamda/imageFolder/']; 
        data_dir = [ 'Lamda/hashCodes/'];
        feature_dir = ['Lamda/features/'];
        load([data_dir 'filenames']); % File names
        load([data_dir 'targets']);   % Labels
        
    case 2
        image_dir =['Barcelona/imageFolder/']; 
        data_dir = ['Barcelona/hashCodes/'];
        feature_dir = ['Barcelona/features/'];
        load([data_dir 'filenames']); % File names
        load([data_dir 'targets']);   % Labels
end


hashCode_index = get(handles.hashCodeSelection_f, 'Value');
switch hashCode_index
           
   case 1
       load([data_dir '/hashCodes_16']); 
       data = hashCodes_16;
       data = max(data,0);
       load([feature_dir '/features_16']); 
       features = features_16;
    case 2
       load([data_dir '/hashCodes_32']); 
       data = hashCodes_32;  
       data = max(data,0);
       load([feature_dir '/features_32']); 
       features = features_32;
    case 3
        load([data_dir '/hashCodes_64']); 
        data = hashCodes_64;
        data = max(data,0);
        load([feature_dir '/features_64']); 
        features = features_64;
   
end


set(handles.QueryName1,'String', filenames);
set(handles.QueryName2,'String', filenames);

handles.filenames = filenames;
handles.targets = targets;
handles.image_dir = image_dir;
handles.data_dir = data_dir;
handles.feature_dir = feature_dir;
handles.data = data;
handles.features = features;

guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function hashCodeSelection_f_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hashCodeSelection_f (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function num_pp_v2_Callback(hObject, eventdata, handles)
% hObject    handle to num_pp_v2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of num_pp_v2 as text
%        str2double(get(hObject,'String')) returns contents of num_pp_v2 as a double


% --- Executes during object creation, after setting all properties.
function num_pp_v2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to num_pp_v2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object deletion, before destroying properties.
function pushbutton11_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function pp_retrieved_Callback(hObject, eventdata, handles)
% hObject    handle to pp_retrieved (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of pp_retrieved as text
%        str2double(get(hObject,'String')) returns contents of pp_retrieved as a double

pp_retrieved = str2double(get(handles.pp_retrieved,'string'));
handles.pp_retrieved = pp_retrieved;

% --- Executes during object creation, after setting all properties.
function pp_retrieved_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pp_retrieved (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function num_pp_Callback(hObject, eventdata, handles)
% hObject    handle to num_pp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of num_pp as text
%        str2double(get(hObject,'String')) returns contents of num_pp as a double


% --- Executes during object creation, after setting all properties.
function num_pp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to num_pp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function acc_box_Callback(hObject, eventdata, handles)
% hObject    handle to acc_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of acc_box as text
%        str2double(get(hObject,'String')) returns contents of acc_box as a double


% --- Executes during object creation, after setting all properties.
function acc_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to acc_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function avg_prec_Callback(hObject, eventdata, handles)
% hObject    handle to avg_prec (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of avg_prec as text
%        str2double(get(hObject,'String')) returns contents of avg_prec as a double


% --- Executes during object creation, after setting all properties.
function avg_prec_CreateFcn(hObject, eventdata, handles)
% hObject    handle to avg_prec (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




function ClusterNum_Callback(hObject, eventdata, handles)
% hObject    handle to ClusterNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ClusterNum as text
%        str2double(get(hObject,'String')) returns contents of ClusterNum as a double


% --- Executes during object creation, after setting all properties.
function ClusterNum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ClusterNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes during object creation, after setting all properties.
function popupmenu6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in rank_euclid.
function rank_euclid_Callback(hObject, eventdata, handles)
% hObject    handle to rank_euclid (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


cla(handles.axes3,'reset');
P = str2double(get(handles.pp_retrieved, 'String'));

queryIndex1 = handles.q1Idx;
queryIndex2 = handles.q2Idx;
features = handles.features;
targets = handles.targets;
data = handles.data;
N = length(handles.filenames);

q_1 = data(queryIndex1,:); 
q_2 = data(queryIndex2,:); 
%q3 = data(queryIndex3,:); 

q1new = repmat(q_1,N,1);
q2new = repmat(q_2,N,1);
%q3new = repmat(q3,N,1);

dist_1 = xor(data, q1new);
dist_2 = xor(data, q2new);
%dist_3 = xor(data, q2new);

hamming_dist1 = sum(dist_1,2);
hamming_dist2 = sum(dist_2,2);
%hamming_dist3 = sum(dist_3,2);

n_hamming_dist1 = mat2gray(hamming_dist1);
n_hamming_dist2 = mat2gray(hamming_dist2);
%n_hamming_dist3 = mat2gray(hamming_dist3);
 
 
X = zeros(2,N);
X(1,:) = hamming_dist1;
X(2,:) = hamming_dist2;
%X(3,:) = n_hamming_dist3;
X = (X)';
input = unique(X, 'rows');

axes(handles.axes3);
hold off; plot(X(:,1),X(:,2),'o','MarkerFaceColor',[0 1 0]); hold on;
set(gca,'FontSize',30); hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
union_of_query_labels = or(targets(queryIndex1, :), targets(queryIndex2, : ));
absolute_union_of_query_labels = nnz(union_of_query_labels);   
%{    
 for e = 1:N        
            MQUR_ALL(e,:) =  nnz( and(targets(e,:) , union_of_query_labels ) ) / absolute_union_of_query_labels ;
            
 end

MQUR_ONE = find(MQUR_ALL == 1);
scatter3(X(MQUR_ONE,1),X(MQUR_ONE,2),X(MQUR_ONE,3),'gs');
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
[K,L] = size(unique(X,'rows'));  %% Number of unique pareto points 
set(handles.num_pp,'String',num2str(K))


% queries location and Optimum point location
q1 = X(queryIndex1,:); 
q2 = X(queryIndex2,:);
line(q1,q2);
Cmn = (q1(:) + q2(:)).'/2;

% https://www.mathworks.com/matlabcentral/answers/44943-properties-of-the-plot-marker-shapes
plot(q1 , q2, 'p' , 'MarkerSize',20,'MarkerFaceColor',[1 0 1]);
plot(Cmn(1), Cmn(2), 'd', 'MarkerSize',20 , 'MarkerFaceColor',[1 0 0]);

Dissim = EuDist2(Cmn, input, 'euclid');
Dissim = Dissim';
[DissimSorted, DissimIndex] = sort(Dissim,'ascend'); % short & get indexes(before shorting)


%%%%%%%%%%%%% Choose First P Shortest distances %%%%%%%%%%%%%%%%%%%%%%%%

DissimIndex_P = DissimIndex(1:P, :); 
Retrieved_PP_indexes  =    ismember(X, input(DissimIndex_P,:),'rows'); 
Retrieved_Items       =    find(Retrieved_PP_indexes); 

plot(X(Retrieved_Items,1),X(Retrieved_Items,2),'ys', 'LineWidth', 2); 
r=DissimSorted(P);
x0=Cmn(1);
y0=Cmn(2);

theta = linspace(0,2*pi,100);
h = plot(x0 + r*cos(theta),y0 + r*sin(theta),'g', 'MarkerSize',5);
%axis equal

%syms x y
%fimplicit((x-x0).^2 + (y-y0).^2 -r^2)
%axis equal

%%%%%%%%%%%%%%%%%%%%  Re-Ranking, rearrange P items by features  %%%%%%%%%%% 

% Add queries to Feature Pareto space, for creating Cmn_f
Retrieved_Items(end+1,:) = queryIndex1;
Retrieved_Items(end+1,:) = queryIndex2;

rtr_idx2_features = features(Retrieved_Items(:,1), :);

f1 = features(queryIndex1,:); 
f2 = features(queryIndex2,:);

dist_f1 = pdist2(f1 , rtr_idx2_features , 'euclid' );
dist_f2 = pdist2(f2 , rtr_idx2_features , 'euclid' );

[M2,B] = size(rtr_idx2_features(:,1));
YY = zeros(2,M2);
YY(1,:) = dist_f1;
YY(2,:) = dist_f2;
YY = (YY)';
for i = 1:M2
    
    plot(YY(:, 1), YY(:,2)  ,'.')    
end


qf1 = YY(end-1,:);  
qf2 = YY(end,:);  
  
Cmn_f = (qf1(:) + qf2(:)).'/2;
plot(Cmn_f(1), Cmn_f(2), 's' , 'LineWidth', 2);

Dissim_f                = EuDist2(YY, Cmn_f, 'euclid');
Dissim_ff               = [Retrieved_Items, Dissim_f];
DissimSorted_f          = sortrows(Dissim_ff, 2);

Retrieved_Items_Ranked  = DissimSorted_f(:,1);

% Now remove last two rows (qf1&qf2) from Retrieved items
Retrieved_Items_Ranked(end) = [];
Retrieved_Items_Ranked(end) = [];

%%%%%%%%%%%%%%%%%%%%%%% Metrics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

predicted_labels  = targets(Retrieved_Items_Ranked,:); % When doing Reranking

[num_R, ~]  = size(Retrieved_Items_Ranked);     
 for e = 1:num_R         
     MQUR_Ranked(e,:) =  nnz( and(predicted_labels(e,:) , union_of_query_labels ) ) / absolute_union_of_query_labels;       
                        
 end
      
for ff = 1:num_R
    if  MQUR_Ranked(ff,:) ~= 1   %%%%%% MQUR !DEN FARKLI OLANLARI SEÇ 
        Retrieved_Items_Ranked(ff,:) = 0;   
        MQUR_Ranked(ff,:) = 0;
    end
end

Retrieved_Items_Ranked( all(~Retrieved_Items_Ranked,2), : ) = [];
MQUR_Ranked( all(~MQUR_Ranked,2), : ) = [];
plot(X(Retrieved_Items_Ranked,1),X(Retrieved_Items_Ranked,2),'bs','LineWidth', 2);
 
 
predicted_labels_ranked  = targets(Retrieved_Items_Ranked,:);
%diff = ismember( predicted_labels_ranked, union_of_query_labels , 'rows' );
if isempty(MQUR_Ranked)
   MQUR_Ranked = 0;
end
  
num_nz = nnz( MQUR_Ranked(:,1) );
s = size(MQUR_Ranked(:,1), 1);
    
for j=1:s;        
    %Cummulative sum of the true-positive elements
    CUMM = cumsum(MQUR_Ranked);          
    Precision_AT_K(j,1) = ( CUMM(j,1)  ) ./ j;              
    Recall_AT_K(j,1) = ( CUMM(j,1)  ) ./ (num_nz); %              
end

avg_Precision = sum(Precision_AT_K(:,1) .* MQUR_Ranked(:,1) ) / num_nz;
avg_Precision(isnan(avg_Precision))=0;
% avg_Precision_OLD = sum(Precision_AT_K(:,1) ) / s;
 acc = num_nz / s;   % accuracy of the best cluster 



set(handles.acc_box,'String',num2str(acc));
set(handles.avg_prec,'String',num2str(avg_Precision));

          
      
      axes(handles.axes11);
      hold off;
      %plot(Recall_AT_K, Precision_AT_K);
      x = linspace(0,s);
      plot( Precision_AT_K )
      ylabel('Precision@k' ,'FontSize', 12)
      xlabel('Number of Rterieved Items' ,'FontSize', 12) 
      hold on;
         

         cla(handles.axes13,'reset');
         cla(handles.axes14,'reset');
         cla(handles.axes15,'reset');
         cla(handles.axes16,'reset');
         cla(handles.axes17,'reset');
         cla(handles.axes18,'reset');         
         cla(handles.axes19,'reset');
         cla(handles.axes20,'reset');
         cla(handles.axes21,'reset');
         cla(handles.axes22,'reset');
         cla(handles.axes23,'reset');
         cla(handles.axes24,'reset');
                 
         
        axes(handles.axes13);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(1,1)}];
        imshow(imread(fname)); 
        set(handles.edit19,'string',num2str( handles.filenames{Retrieved_Items_Ranked(1,1)}));
        axis image
      
        axes(handles.axes14);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(2,1)}];
        imshow(imread(fname)); 
        set(handles.edit20,'string',num2str(  handles.filenames{Retrieved_Items_Ranked(2,1)}));
        axis image
        
        axes(handles.axes15);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(3,1)}];
        imshow(imread(fname));
        set(handles.edit21,'string',num2str( handles.filenames{Retrieved_Items_Ranked(3,1)}));
        axis image
        
        axes(handles.axes16);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(4,1)}];
        imshow(imread(fname)); 
        set(handles.edit22,'string',num2str( handles.filenames{Retrieved_Items_Ranked(4,1)}));
        axis image
        
      
        axes(handles.axes17);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(5,1)}];
        imshow(imread(fname)); 
        set(handles.edit23,'string',num2str( handles.filenames{Retrieved_Items_Ranked(5,1)}));
        axis image
        
       
        axes(handles.axes18);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(6,1)}];
        imshow(imread(fname)); 
        set(handles.edit24,'string',num2str( handles.filenames{Retrieved_Items_Ranked(6,1)}));
        axis image
        
       
        axes(handles.axes19);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(7,1)}];
        imshow(imread(fname)); 
        set(handles.edit25,'string',num2str( handles.filenames{Retrieved_Items_Ranked(7,1)}));
        axis image
        
      
        axes(handles.axes20);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(8,1)}];
        imshow(imread(fname)); 
        set(handles.edit26,'string',num2str( handles.filenames{Retrieved_Items_Ranked(8,1)}));
        axis image
        
     
        axes(handles.axes21);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(9,1)}];
        imshow(imread(fname)); 
        set(handles.edit27,'string',num2str( handles.filenames{Retrieved_Items_Ranked(9,1)}));
        axis image
        
     
        axes(handles.axes22);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(10,1)}];
        imshow(imread(fname)); 
        set(handles.edit28,'string',num2str( handles.filenames{Retrieved_Items_Ranked(10,1)}));
        axis image
        
       
        axes(handles.axes23);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(11,1)}];
        imshow(imread(fname)); 
        set(handles.edit29,'string',num2str( handles.filenames{Retrieved_Items_Ranked(11,1)}));
        axis image
        
    
        axes(handles.axes24);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(12,1)}];
        imshow(imread(fname)); 
        set(handles.edit30,'string',num2str( handles.filenames{Retrieved_Items_Ranked(12,1)}));
        axis image
        
            
              
            
      
 guidata(hObject, handles);


 


% --- Executes on button press in rank_emr.
function rank_emr_Callback(hObject, eventdata, handles)
% hObject    handle to rank_emr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

cla(handles.axes3,'reset');
P = str2double(get(handles.pp_retrieved, 'String'));
p = str2double(get(handles.num_landmarks, 'String'));


queryIndex1 = handles.q1Idx;
queryIndex2 = handles.q2Idx;
features = handles.features;
targets = handles.targets;
data = handles.data;
N = length(handles.filenames);

q_1 = data(queryIndex1,:); 
q_2 = data(queryIndex2,:); 
%q3 = data(queryIndex3,:); 

q1new = repmat(q_1,N,1);
q2new = repmat(q_2,N,1);
%q3new = repmat(q3,N,1);

dist_1 = xor(data, q1new);
dist_2 = xor(data, q2new);
%dist_3 = xor(data, q2new);

hamming_dist1 = sum(dist_1,2);
hamming_dist2 = sum(dist_2,2);
%hamming_dist3 = sum(dist_3,2);

n_hamming_dist1 = mat2gray(hamming_dist1);
n_hamming_dist2 = mat2gray(hamming_dist2);
%n_hamming_dist3 = mat2gray(hamming_dist3);
 
 
X = zeros(2,N);
X(1,:) = hamming_dist1;
X(2,:) = hamming_dist2;
%X(3,:) = n_hamming_dist3;
X = (X)';
input = unique(X, 'rows');

axes(handles.axes3);
hold off; plot(X(:,1),X(:,2),'o','MarkerFaceColor',[0 1 0]); hold on;
set(gca,'FontSize',30); hold on;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
union_of_query_labels = or(targets(queryIndex1, :), targets(queryIndex2, : ));
absolute_union_of_query_labels = nnz(union_of_query_labels);   
%{    
 for e = 1:N        
            MQUR_ALL(e,:) =  nnz( and(targets(e,:) , union_of_query_labels ) ) / absolute_union_of_query_labels ;
            
 end

MQUR_ONE = find(MQUR_ALL == 1);
scatter3(X(MQUR_ONE,1),X(MQUR_ONE,2),X(MQUR_ONE,3),'gs');
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 

[K,L] = size(unique(X,'rows'));  %% Number of unique pareto points 
set(handles.num_pp,'String',num2str(K))


% queries location and Optimum point location
q1 = X(queryIndex1,:); 
q2 = X(queryIndex2,:);
line(q1,q2);
Cmn = (q1(:) + q2(:)).'/2;

% https://www.mathworks.com/matlabcentral/answers/44943-properties-of-the-plot-marker-shapes
plot(q1 , q2, 'p' , 'MarkerSize',20,'MarkerFaceColor',[1 0 1]);
plot(Cmn(1), Cmn(2), 'd', 'MarkerSize',20 , 'MarkerFaceColor',[1 0 0]);

%%%%%%%%%%%%%%%%%%%%%%%%%%% Add Cmn in the Pareto Space and set its rank as 1 %%%%%%%%%%%%%%%
input(end+1,:) =  Cmn;
nSmp = size(input,1);
y1 = zeros(nSmp, 1);
y1(end) = 1; %  Set The Rank of Cmn (Query) as 1 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EMR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.p = p;
%opts.p=100; 	% the number of landmarks picked (default 1000)
opts.r=5;  	% the number of nearest landmarks for representation (default 5)
opts.a=0.99; 	% weight in manifold ranking, score = (I - aS)^(-1)y, default  0.99
kmMaxIter = 5;
kmNumRep = 1;

 
[dump,landmarks]=litekmeans(input ,opts.p,'MaxIter',kmMaxIter,'Replicates',kmNumRep); 

D = EuDist2( input ,landmarks);
dump = zeros(nSmp,opts.r); % dump was a 549x1 vector, but after this it is 549x2 vector !
idx = dump;

for i = 1:opts.r
    [dump(:,i),idx(:,i)] = min(D,[],2);
    temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
    D(temp) = 1e100;
end

dump = bsxfun(@rdivide,dump,dump(:,opts.r));
dump = 0.75 * (1 - dump.^2);
Gsdx = dump;
Gidx = repmat([1:nSmp]',1,opts.r);
Gjdx = idx;
Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,opts.p);

model.Z = Z';		

% Efficient Ranking
feaSum = full(sum(Z,1));
D = Z*feaSum';
D = max(D, 1e-12);
D = D.^(-.5);
H = spdiags(D,0,nSmp,nSmp)*Z;

C = speye(opts.p);
A = H'*H-(1/opts.a)*C;

%%%%%%%%%%%%%%%%%%%%%%%%%%%% EMR Ranking & Retrieval by Hash Codes %%%%%%%
simEMR = EMRscore(H ,A, y1);
Dissim = 1-simEMR;                           % Dissimilarity
[DissimSorted, DissimIndex] = sort(Dissim,'ascend');
%input(end,:) =  [];                          % Remove Cmn from space
DissimIndex  = DissimIndex(2:end, :);         % Remove Cmn's Index

DissimIndex_P         =    DissimIndex(1:P, :); 
Retrieved_PP_indexes  =    ismember(X, input(DissimIndex_P,:),'rows'); 
Retrieved_Items       =    find(Retrieved_PP_indexes); 

plot(X(Retrieved_Items,1),X(Retrieved_Items,2),'ys', 'LineWidth', 2); 


%%%%%%%%%%%%%%%%%%%%  Re-Ranking, rearrange P items by features  %%%%%%%%%%

% Add queries to Feature Pareto space, for creating Cmn_f
Retrieved_Items(end+1,:) = queryIndex1;
Retrieved_Items(end+1,:) = queryIndex2;

rtr_idx2_features = features(Retrieved_Items(:,1), :);


f1 = features(queryIndex1,:); 
f2 = features(queryIndex2,:);

dist_f1 = pdist2(f1 , rtr_idx2_features , 'euclid' );
dist_f2 = pdist2(f2 , rtr_idx2_features , 'euclid' );

[M2,B] = size(rtr_idx2_features(:,1));
YY = zeros(2,M2);
YY(1,:) = dist_f1;
YY(2,:) = dist_f2;
YY = (YY)';
for i = 1:M2
    
    plot(YY(:, 1), YY(:,2)  ,'.')    
end


qf1 = YY(end-1,:);  
qf2 = YY(end,:);  
  
Cmn_f = (qf1(:) + qf2(:)).'/2;
plot(Cmn_f(1), Cmn_f(2), 's' , 'LineWidth', 2);

Dissim_f                = EuDist2(YY, Cmn_f, 'euclid');
Dissim_ff               = [Retrieved_Items, Dissim_f];
DissimSorted_f          = sortrows(Dissim_ff, 2);

Retrieved_Items_Ranked  = DissimSorted_f(:,1);

% Now remove last two rows (qf1&qf2) from Retrieved items
Retrieved_Items_Ranked(end) = [];
Retrieved_Items_Ranked(end) = [];

%%%%%%%%%%%%%%%%%%%%%%% Metrics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

predicted_labels  = targets(Retrieved_Items_Ranked,:); % When doing Reranking

[num_R, ~]  = size(Retrieved_Items_Ranked);     
 for e = 1:num_R         
     MQUR_Ranked(e,:) =  nnz( and(predicted_labels(e,:) , union_of_query_labels ) ) / absolute_union_of_query_labels;       
                        
 end
      
for ff = 1:num_R
    if  MQUR_Ranked(ff,:) ~= 1   %%%%%% MQUR !DEN FARKLI OLANLARI SEÇ 
        Retrieved_Items_Ranked(ff,:) = 0;   
        MQUR_Ranked(ff,:) = 0;
    end
end

Retrieved_Items_Ranked( all(~Retrieved_Items_Ranked,2), : ) = [];
MQUR_Ranked( all(~MQUR_Ranked,2), : ) = [];
plot(X(Retrieved_Items_Ranked,1),X(Retrieved_Items_Ranked,2),'bs','LineWidth', 2);
 
 
predicted_labels_ranked  = targets(Retrieved_Items_Ranked,:);
%diff = ismember( predicted_labels_ranked, union_of_query_labels , 'rows' );
if isempty(MQUR_Ranked)
   MQUR_Ranked = 0;
end
  
num_nz = nnz( MQUR_Ranked(:,1) );
s = size(MQUR_Ranked(:,1), 1);
    
for j=1:s;        
    %Cummulative sum of the true-positive elements
    CUMM = cumsum(MQUR_Ranked);          
    Precision_AT_K(j,1) = ( CUMM(j,1)  ) ./ j;              
    Recall_AT_K(j,1) = ( CUMM(j,1)  ) ./ (num_nz); %              
end

avg_Precision = sum(Precision_AT_K(:,1) .* MQUR_Ranked(:,1) ) / num_nz;
avg_Precision(isnan(avg_Precision))=0;
% avg_Precision_OLD = sum(Precision_AT_K(:,1) ) / s;
 acc = num_nz / s;   % accuracy of the best cluster 


set(handles.acc_box,'String',num2str(acc));
set(handles.avg_prec,'String',num2str(avg_Precision));

          
      
      axes(handles.axes11);
      hold off;
      %plot(Recall_AT_K, Precision_AT_K);
      x = linspace(0,s);
      plot( Precision_AT_K )
      ylabel('Precision@k' ,'FontSize', 12)
      xlabel('Number of Rterieved Items' ,'FontSize', 12) 
      hold on;
         

         cla(handles.axes13,'reset');
         cla(handles.axes14,'reset');
         cla(handles.axes15,'reset');
         cla(handles.axes16,'reset');
         cla(handles.axes17,'reset');
         cla(handles.axes18,'reset');         
         cla(handles.axes19,'reset');
         cla(handles.axes20,'reset');
         cla(handles.axes21,'reset');
         cla(handles.axes22,'reset');
         cla(handles.axes23,'reset');
         cla(handles.axes24,'reset');
                  
         
        axes(handles.axes13);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(1,1)}];
        imshow(imread(fname)); 
        set(handles.edit19,'string',num2str( handles.filenames{Retrieved_Items_Ranked(1,1)}));
        axis image
      
        axes(handles.axes14);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(2,1)}];
        imshow(imread(fname)); 
        set(handles.edit20,'string',num2str(  handles.filenames{Retrieved_Items_Ranked(2,1)}));
        axis image
        
        axes(handles.axes15);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(3,1)}];
        imshow(imread(fname));
        set(handles.edit21,'string',num2str( handles.filenames{Retrieved_Items_Ranked(3,1)}));
        axis image
        
        axes(handles.axes16);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(4,1)}];
        imshow(imread(fname)); 
        set(handles.edit22,'string',num2str( handles.filenames{Retrieved_Items_Ranked(4,1)}));
        axis image
        
      
        axes(handles.axes17);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(5,1)}];
        imshow(imread(fname)); 
        set(handles.edit23,'string',num2str( handles.filenames{Retrieved_Items_Ranked(5,1)}));
        axis image
        
       
        axes(handles.axes18);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(6,1)}];
        imshow(imread(fname)); 
        set(handles.edit24,'string',num2str( handles.filenames{Retrieved_Items_Ranked(6,1)}));
        axis image
        
       
        axes(handles.axes19);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(7,1)}];
        imshow(imread(fname)); 
        set(handles.edit25,'string',num2str( handles.filenames{Retrieved_Items_Ranked(7,1)}));
        axis image
        
      
        axes(handles.axes20);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(8,1)}];
        imshow(imread(fname)); 
        set(handles.edit26,'string',num2str( handles.filenames{Retrieved_Items_Ranked(8,1)}));
        axis image
        
     
        axes(handles.axes21);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(9,1)}];
        imshow(imread(fname)); 
        set(handles.edit27,'string',num2str( handles.filenames{Retrieved_Items_Ranked(9,1)}));
        axis image
        
     
        axes(handles.axes22);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(10,1)}];
        imshow(imread(fname)); 
        set(handles.edit28,'string',num2str( handles.filenames{Retrieved_Items_Ranked(10,1)}));
        axis image
        
       
        axes(handles.axes23);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(11,1)}];
        imshow(imread(fname)); 
        set(handles.edit29,'string',num2str( handles.filenames{Retrieved_Items_Ranked(11,1)}));
        axis image
        
    
        axes(handles.axes24);
        fname = [handles.image_dir handles.filenames{Retrieved_Items_Ranked(12,1)}];
        imshow(imread(fname)); 
        set(handles.edit30,'string',num2str( handles.filenames{Retrieved_Items_Ranked(12,1)}));
        axis image
        
            
              
            
      
 guidata(hObject, handles);



function num_landmarks_Callback(hObject, eventdata, handles)
% hObject    handle to num_landmarks (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of num_landmarks as text
%        str2double(get(hObject,'String')) returns contents of num_landmarks as a double
num_landmarks = str2double(get(handles.num_landmarks,'string'));
handles.num_landmarks = num_landmarks;

% --- Executes during object creation, after setting all properties.
function num_landmarks_CreateFcn(hObject, eventdata, handles)
% hObject    handle to num_landmarks (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
