import base64, datetime, io, os, PyPDF2, pytesseract, shutil, time
from dash import Dash, dcc, html, Input, Output, State, callback
from db_build import run_db_build
from llm import query
from pdf2image import convert_from_path

app = Dash(__name__)

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

app.layout = html.Div([
    dcc.Store(
        id='memory', 
        data={
            'db_path':'assets/PublicWaterMassMailing',
            'transcribed':True,
            'indexed':True}),
    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                dcc.Upload(
                    id='upload',
                    children=
                        html.Div(['Drag and Drop or ', html.A('click to select PDF')]),
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
                    html.Button('Index', id='index-btn', style={'display':'inline-block', 'margin': '5px 10px', 'width':'99px', 'height':'40px', 'padding':'0px'}),
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
                        placeholder='Ask a question like "Who is the Governor?" \nPress query to submit', 
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
        html.Div(id='filename-hidden', style={'display':'none'}, children='PublicWaterMassMailing.pdf'),
        html.Div(id='db-path-hidden', style={'display':'none'}),
        html.Div(id='delete-hidden', style={'display':'none'})
    ])
])

@callback(Output('delete-hidden', 'children'),
          Input('delete-btn', 'n_clicks'),
          prevent_initial_call=True)
def delete_btn(n_clicks):
    clear_files('assets/temp/')
    clear_dir('assets/db/')
    return [None]

def clear_files(dir, sample_fname='PublicWaterMassMailing.pdf'):
    for fname in os.listdir(dir):
        if sample_fname != fname:
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
           State('db-path-hidden', 'children')],
           prevent_initial_callback=True)
def llm_query(n_clicks, qstring, db_path):
    if n_clicks:
        if qstring and db_path:
            print(f'Querying: "{qstring}"')
            response = query(qstring, db_path)
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
    

def parse_contents(contents, filename, date):
    if filename[-4:] == '.pdf':
        return parse_pdf(contents, filename, date)
    else:
        return parse_unsupported(filename)
        # return parse_img(contents, filename, date)

def parse_unsupported(filename):
    return html.Div(html.H5(f"{filename} is unsupported; only .pdf is currently supported"))
    
def parse_img(contents, filename, date):
    content_type, content_data = contents.split(',')
    fh = open(f"assets/temp/{filename}", "wb")
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

def parse_pdf(contents, filename, date):
    content_type, content_data = contents.split(',')
    if '_transcribed.pdf' not in filename:
        fh = open(f"assets/temp/{filename}", "wb")
        fh.write(base64.b64decode(content_data))
        fh.close()

@callback([Output('filename-hidden', 'children', allow_duplicate=True),
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

@callback([Output('output-upload', 'children'),
           Output('filename-visible','children')],
          Input('filename-hidden', 'children'))
def update_preview(filename):
    output_upload =  html.Div([
            html.Iframe(src=f"assets/temp/{filename}", style = {'width':'98%', 'height':'888px', 'margin':'0px 10px'})
            ])
    filename_visible = html.Div(
        html.H6(filename))
    return output_upload, filename_visible
    
@callback([Output('transcribed', 'children')],
           [Input('filename-hidden', 'children')])
def update_status(filename):
    transcribed = None
    if '_transcribed.pdf' in filename:
        transcribed = f"OCR applied to {filename.replace('_transcribed.pdf','')} successfully"
    transcribed = transcribed or 'Not yet transcribed: Only necessary when the .pdf is scanned'
    
    return [html.Div([html.P(transcribed)])]

def base_filename(filename):
    return filename.replace('_transcribed.pdf', '').replace('.pdf', '')

def get_db_path(filename):
    return 'assets/db/' + base_filename(filename) + '/'

def db_exists(db_path):
    return os.path.exists(db_path+'index.faiss') and os.path.exists(db_path+'index.pkl')

@callback([Output('transcribed', 'children', allow_duplicate=True),
           Output('filename-hidden', 'children', allow_duplicate=True)],
           [Input('transcribe-btn', 'n_clicks')],
           [State('filename-hidden', 'children')],
           prevent_initial_call=True)
def transcribe(n_clicks, filename):
    filepath = 'assets/temp/' + filename
    print(f'Filename: {filename}\nFilepath: {filepath}')
    if n_clicks >= 1:
        if '_transcribed.pdf' in filename:
            transcribed = 'Transcribed: True'
        elif os.path.exists(filepath.replace('.pdf', '_transcribed.pdf')):
            transcribed = 'Transcribed: True'
            filename = filename.replace('.pdf', '_transcribed.pdf')
            print(f'{filename} previously transcribed, loading temp file')
        else:
            try:
                images = convert_from_path(filepath)
                pdf_writer = PyPDF2.PdfWriter()
                for i, image in enumerate(images):
                    print(f'Transcribing page {i+1} of {len(images)}')
                    page = pytesseract.image_to_pdf_or_hocr(image)
                    pdf = PyPDF2.PdfReader(io.BytesIO(page))
                    pdf_writer.add_page(pdf.pages[0])
                print(f'{len(images)} pages transcribed')
                filename = filename.replace('.pdf', '_transcribed.pdf')
                filepath = filepath.replace('.pdf', '_transcribed.pdf')
                with open(filepath, 'wb') as f:
                    pdf_writer.write(f)
                transcribed = 'Transcribed: True'
            except:
                transcribed = 'Transcribed: Failed to transcribe file'
    return html.Div([html.H6(transcribed)]), filename

@callback([Output('db-path-hidden', 'children'),
           Output('indexed', 'children')],
          [Input('index-btn', 'n_clicks')],
          [State('filename-hidden', 'children')])
def index(n_clicks, filename):
    n_clicks = n_clicks or 0
    db_path, text = None, None
    if n_clicks >= 1:
        db_path = get_db_path(filename)
        if db_exists(db_path):
            text = f'Index previously built for {filename}'
        try:
            print(f'Building index for {filename}')
            run_db_build(filename, 'assets/temp/', db_path)
            text = f'Index built for {filename}'
        except Exception as E:
            text = f'Failed to build index for {filename}'
            print(E)
    text = text or 'File not yet indexed'
    return [db_path, html.Div([html.P(text)])]

if __name__ == '__main__':
    app.run(debug=False, dev_tools_hot_reload=False)
    # app.run(debug=True)