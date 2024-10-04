import streamlit as st
import pandas as pd

# ------ PART 1 ------

df = pd.DataFrame(
    [
       {"command": "st.selectbox", "rating": 4, "is_widget": True},
       {"command": "st.balloons", "rating": 5, "is_widget": False},
       {"command": "st.time_input", "rating": 3, "is_widget": True},
   ]
)

# Display text
st.text('Fixed width text')
st.markdown('_**Markdown**_') # see #*
st.caption('Balloons. Hundreds of them...')
st.latex(r''' e^{i\pi} + 1 = 0 ''')
st.write('Most objects') # df, err, func, keras!
st.write(['st', 'is <', 3]) # see *

# TITLE
st.title('Problematic Internet Use - Group 20 Project Proposal')



st.header('Introduction')
st.subheader('Literature Review')
url1 = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7330165/"
st.write('[This paper](%s) explores the impacts of physical fitness on reducing internet addiction. The paper finds that physical activity can cause key changes in brain chemistry concerning internet addiction. Specifically, they find that fitness can improve things like attention span and dopamine production, which can be correlated with addiction.' % url)
st.code('for i in range(8): foo()')

# * optional kwarg unsafe_allow_html = True

# Interactive widgets
st.button('Hit me')
st.data_editor(df)
st.checkbox('Check me out')
st.radio('Pick one:', ['nose','ear'])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiselect', [1,2,3])
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
st.file_uploader('File uploader')

# -- add download button (start) --
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',
)
# -- add download button (end) --



# ------ PART 2 ------

data = pd.read_csv("employees.csv")

# Display Data
st.dataframe(data)
st.table(data.iloc[0:10])
st.json({'foo':'bar','fu':'ba'})
st.metric('My metric', 42, 2)

# Media
st.image('./smile.png')
