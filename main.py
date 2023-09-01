import base64, datetime, io, os, PyPDF2, pytesseract, shutil, time
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_daq as daq
from src.db_build import run_db_build
from src.llm import query, build_llm
from pdf2image import convert_from_path

app = Dash(__name__)
LLM = build_llm()

sample_filename = 'PublicWaterMassMailing.pdf'
db_dir = 'db/'
files_dir = 'assets/temp/'
transcribed_dir = 'assets/temp/tx/'

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

# # # start Misc. Helpers
def base_filename(filename):
    return filename.replace('.pdf', '')

def get_db_path(filename):
    return db_dir + base_filename(filename) + '/'

def get_transcribed_path(filename):
    return db_dir + base_filename(filename) + '/'

def db_exists(db_path):
    return os.path.exists(db_path+'index.faiss') and os.path.exists(db_path+'index.pkl')

def transcribe_pdf(filepath, transcribed_filepath):
    images = convert_from_path(filepath)
    pdf_writer = PyPDF2.PdfWriter()
    for i, image in enumerate(images):
        print(f'Transcribing page {i+1} of {len(images)}')
        page = pytesseract.image_to_pdf_or_hocr(image)
        pdf = PyPDF2.PdfReader(io.BytesIO(page))
        pdf_writer.add_page(pdf.pages[0])
    print(f'{len(images)} pages transcribed')
    with open(transcribed_filepath, 'wb') as f:
        pdf_writer.write(f)

def is_transcribed(filename):
    return os.path.exists(transcribed_dir + filename)
# # # end Misc. Helpers

app.layout = html.Div([
    dcc.Store(
        id='memory', 
        data={
            'db_path':f'{files_dir}{sample_filename}',
            'transcribed':True,
            'indexed':True}),
    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                dcc.Upload(
                    id='upload',
                    children=
                        html.Div([html.A('Upload a new PDF')]),
                    style={'width': '100%', 'height': '50px', 'lineHeight': '50px',
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                        'textAlign': 'center', 'margin': '10px', 'display':'inline-block'},
                    multiple=False)], 
                style={'display':'inline-block', 'width':'50%'}),
            html.Button('Delete Temporary Files', id='delete-btn', style={'display':'inline-block', 'margin': '10px 5px', 'height':'50px', 'float':'right', 'backgroundColor':'red', 'color':'white'}),
            html.Div(id='output-upload')],
            style={'width': '50%', 'display': 'inline-block', 'vertical-align':'top'}),
        html.Div(children=[
            html.Div(children=[
                html.Div(id='filename-visible', style={'margin': '20px 10px'}),
                html.Div(children=[
                    html.Button('Transcribe', id='transcribe-btn', style={'display':'inline-block', 'margin': '5px 10px', 'width':'99px', 'height':'40px', 'padding':'0px'}),
                    html.Div(children=[
                        dcc.Loading(id='transcribed',
                                    style={'display':'inline-flex'})],
                        style={'display':'inline-flex'})
                    ]),
                html.Div(children=[
                    html.Div(html.P('Token database chunk parameters'), style={'margin':'0px'}),
                    daq.NumericInput(
                        id='chunk-size-input',
                        label='Size',
                        labelPosition='top',
                        size=100,
                        value=250,
                        min=100,
                        max=2000,
                        style={'display':'inline-flex', 'margin-right':'5px'}
                    ),
                    daq.NumericInput(
                        id='chunk-overlap-input',
                        label='Overlap',
                        labelPosition='top',
                        size=100,
                        value=50,
                        min=20,
                        max=500,
                        style={'display':'inline-flex'}
                    )
                ],
                style={'margin':'40px 10px 0px 10px', 'display':'none'}),
                html.Div(children=[
                    html.Button('Index', id='index-btn', style={'display':'inline-block', 'margin':'5px 10px 40px 10px', 'width':'99px', 'height':'40px', 'padding':'0px'}),
                    html.Div(children=[
                        dcc.Loading(html.Div(id='indexed'))], 
                        style={'display':'inline-flex'})
                    ]),
                html.Div(children=[
                    html.Button('Query', 
                                id='query-btn', 
                                style={'display':'inline-block', 'margin': '5px 10px', 'width':'99px', 'height':'66px', 'padding':'0px', 'verticalAlign':'top'}),
                    dcc.Textarea(
                        id='input-query', 
                        placeholder='Enter question here and press query to submit', 
                        style={'margin':'5px 0px', 'width':'77%', 'height':'60px','display':'inline-block'})
                    ]),
                dcc.Loading(
                    children=[
                        html.Div(id='output-query', 
                                 style={'margin':'10px 20px'})],
                    type='default')
                ])
            ], 
            style={"width": '50%', 'display': 'inline-block', 'vertical-align':'top'})
    ]),
    html.Div(children=[
        html.Div(id='filename-hidden', style={'display':'none'}, children=sample_filename),
        html.Div(id='filepath-hidden', style={'display':'none'}, children=f'{files_dir}{sample_filename}'),
        html.Div(id='delete-hidden', style={'display':'none'})
    ])
])

@callback(Output('delete-hidden', 'children'),
          Input('delete-btn', 'n_clicks'),
          State('filename-hidden', 'children'),
          prevent_initial_call=True)
def delete_btn(n_clicks, cur_filename):
    print('Clearing temp uploaded and transcribed files')
    clear_files(files_dir, cur_filename); clear_files(transcribed_dir, cur_filename); 
    print('Clearing temp vector databases')
    clear_dir(db_dir)
    return [None]

def clear_files(dir, cur_filename):
    for fname in os.listdir(dir):
        if fname != sample_filename and fname == cur_filename:
            try:
                os.remove(dir + fname)
                print(f'Deleted {fname} from {dir}')
            except Exception as E:
                print(f'Failed to delete {fname} from {dir}')
                print(E)

def clear_dir(dir):
    for fname in os.listdir(dir):
        try:
            shutil.rmtree(dir + fname)
            print(f'Deleted {fname} from {dir}')
        except Exception as E:
            print(f'Failed to delete {fname} from {dir}')
            print(E)

@callback(Output('output-query', 'children'),
          [Input('query-btn', 'n_clicks'),
           State('input-query', 'value'),
           State('filename-hidden', 'children')],
           prevent_initial_callback=True)
def llm_query(n_clicks, qstring, filename):
    db_path = get_db_path(filename)
    if n_clicks and db_exists(db_path):
        if qstring and db_path:
            print(f'Querying: "{qstring}"')
            response = query(qstring, db_path, LLM)
            return html.Div(children=[
                html.Div(children=[
                    html.B('Question:', style={'display':'inline-block', 'margin-right':'10px'}),
                    html.P(f'{response["query"]}', style={'display':'inline-block'})
                ]),
                html.Div(children=[
                    html.B('Answer: ', style={'display':'inline-block', 'margin-right':'10px'}),
                    html.P(f'{response["result"]}', style={'display':'inline-block'})
                ]),
                html.Div(children=[
                    html.B('Time: ', style={'display':'inline-block', 'margin-right':'10px'}),
                    html.P(f'{response["time"]} seconds', style={'display':'inline-block'})
                ]),
                html.Div(html.B('Citations:')),
                html.Div(
                    children=[
                        html.P([f'Page: {doc.metadata["page"]+1}',html.Br(), f'{doc.page_content}'], style={'margin':'20px 0px'}) for doc in response['source_documents']
                    ]
                ),
            ])
        elif qstring:
            return html.Div(html.P('Index not yet built'))
    return html.Div(html.P('Result will appear here'))

@callback([Output('filename-hidden', 'children'),
           Output('transcribe-btn', 'n_clicks'),
           Output('index-btn', 'n_clicks')],
           [Input('upload', 'contents'),
           State('upload', 'filename'),
           State('upload', 'last_modified')],
           prevent_initial_call=True)
def update_output(contents, filename, date):
    if contents is not None:
        parse_contents(contents, filename, date)
        return filename, 0, 0

def parse_contents(contents, filename, date):
    if filename[-4:] == '.pdf':
        return parse_pdf(contents, filename, date)
    else:
        return parse_unsupported(filename)

def parse_pdf(contents, filename, date):
    content_type, content_data = contents.split(',')
    if is_transcribed(filename):
        filepath = transcribed_dir + filename
    else:
        filepath = files_dir + filename
    fh = open(filepath, 'wb')
    fh.write(base64.b64decode(content_data))
    fh.close()

def parse_unsupported(filename):
    return html.Div(html.H5(f"{filename} is unsupported; only .pdf is currently supported"))
    
def parse_img(contents, filename, date):
    content_type, content_data = contents.split(',')
    fh = open(files_dir + f'/{filename}', 'wb')
    fh.write(base64.b64decode(content_data))
    fh.close()

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@callback([Output('output-upload', 'children'),
           Output('filename-visible','children')],
          Input('filepath-hidden', 'children'),
          Input('filename-hidden', 'children'))
def update_preview(filepath, filename):
    print('Updating pdf preview')
    output_upload =  html.Div([
            html.Iframe(src=filepath, style = {'width':'98%', 'height':'888px', 'margin':'0px 10px'})
        ])
    filename_visible = html.Div(
        html.H6(filename))
    return output_upload, filename_visible

@callback([Output('transcribed', 'children'),
           Output('filepath-hidden', 'children')],
           [Input('transcribe-btn', 'n_clicks')],
           [State('filename-hidden', 'children')])
def transcribe(n_clicks, filename):
    n_clicks = n_clicks or 0
    filepath = files_dir + filename
    transcribed_filepath = transcribed_dir + filename
    print(f'Filename: {filename}\nFilepath: {filepath}')
    if n_clicks >= 1:
        if os.path.exists(transcribed_filepath):
            text = f'Success: {filename} previously transcribed, cached file loaded'
            print(f'{filename} previously transcribed, loading cached file')
        else:
            try:
                transcribe_pdf(filepath, transcribed_filepath)
                text = f"Success: OCR applied to {filename} successfully"
            except Exception as E:
                print(E)
                text = f'Failed to transcribe file: {E}'
        filepath = transcribed_filepath
    else:
        text = 'Not yet transcribed: only necessary when the .pdf is scanned'
    return html.Div([html.P(text)]), filepath

@callback([Output('indexed', 'children')],
          [Input('index-btn', 'n_clicks')],
          [State('filename-hidden', 'children'),
           State('chunk-size-input', 'value'),
           State('chunk-overlap-input', 'value')])
def index(n_clicks, filename, chunk_size, chunk_overlap):
    n_clicks = n_clicks or 0
    db_path, text = None, None
    if n_clicks >= 1:
        db_path = get_db_path(filename)
        if not db_exists(db_path):
            try:
                print(f'Building index for {filename}')
                if is_transcribed(filename):
                    run_db_build(filename, transcribed_dir, db_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                else:
                    run_db_build(filename, files_dir, db_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                text = f'Success: Index built for {filename}'
            except Exception as E:
                text = f'Error: Failed to build index for {filename}'
                print(E)
        else:
            text = f'Index previously built for {filename}'
    text = text or 'Not yet indexed: .pdf must be digital native or transcribed first'
    return [html.Div([html.P(text)])]

if __name__ == '__main__':
    # app.run(debug=True, dev_tools_hot_reload=True)
    app.run(debug=False, dev_tools_hot_reload=False)
