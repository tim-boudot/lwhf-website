import streamlit as st
import plotly.express as px
import pandas as pd
import time
import requests
import stqdm
import os, json

#predit_url = ('https://lwhf-edxf3vliba-ew.a.run.app/predict')
current_url = ('https://lwhf7-edxf3vliba-ew.a.run.app/current_portfolio')

api_cache_folder = 'api_cache_7'


st.markdown('''# Want a money-making portfolio to invest today?''')

options = st.multiselect(
    "Which asset classes would you like to include in your portfolio",
    ["Equities", "Bonds", "Real Estate", "Bitcoin"],
    ["Equities", "Bonds", "Real Estate", "Bitcoin"])

st.write(" ")
st.write(" ")


if st.button("Let's create that portfolio!", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):

    end_point = 'current_portfolio'
    with st.status("Building Portfolio...", expanded=True) as status:
        st.write("Downloading data...")
        time.sleep(1)
        st.write("Running model...")

        params = {
            'as_of_date':'2024-05-27'
        }

        full_path = os.path.join(api_cache_folder, end_point)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        as_of_date = params['as_of_date']
        filename = f'{as_of_date}.json'
        json_full_path = os.path.join(full_path, filename)

        if os.path.exists(json_full_path):
            st.info("Pre-trained model found. Model predicting...")
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.001)  # Adjust the sleep time to make the progress bar fill up in 3 seconds
                progress_bar.progress(percent_complete + 1)

            # read the json file as a dictionary
            with open(json_full_path, 'r') as file:
                res = json.load(file)
        else:
            st.warning("Model training will take approximately 3 minutes. Please wait...")
            req = requests.get(current_url, params)
            res = req.json()
            with open(json_full_path, 'w') as f:
                json.dump(res, f)

        exp_ret = res[0]
        exp_var = res[1]
        weights = res[2]['weights']
        weights = {k:v for k,v in weights.items() if v!=0}

        dicto_3 = {'Stocks':weights.keys(), 'Values':weights.values()}
        fig = px.pie(dicto_3, values='Values', names='Stocks')

        st.write("Optimizing portfolio...")
        time.sleep(3)
        status.update(label='Portfolio ready!', state='complete', expanded=False)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Expected Return:')
        st.write(f'{round(exp_ret*100,2)}%')

    with col2:
        st.subheader('Expected Variance:')
        st.write(f'{round(exp_var*100,2)}%')

    st.write(" ")
    st.write(" ")
    st.plotly_chart(fig, use_container_width=True)






    # progress_text = "Operation in progress. Please wait."
    # my_bar = st.progress(0, text=progress_text)


    # for percent_complete in range(100):
    #     time.sleep(0.01)
    #     my_bar.progress(percent_complete + 1, text=progress_text)

    # time.sleep(1)
    # my_bar.empty()

def rain(
    emoji: str,
    font_size: int = 64,
    falling_speed: int = 5,
    animation_length: str = "infinite",
):
    """
    Creates a CSS animation where input emoji falls from top to bottom of the screen.

    Args:
        emoji (str): Emoji
        font_size (int, optional): Font size. Defaults to 64.
        falling_speed (int, optional): Speed at which the emoji 'falls'. Defaults to 5.
        animation_length (Union[int, str], optional): Length of the animation. Defaults to "infinite".
    """

    if isinstance(animation_length, int):
        animation_length = f"{animation_length}"

    st.write(
        f"""
    <style>

    body {{
    background: gray;
    }}

    .emoji {{
    color: #777;
    font-size: {font_size}px;
    font-family: Arial;
    // text-shadow: 0 0 5px #000;
    }}

    ///*delete for no hover-effect*/
    //.emoji:hover {{
    //  font-size: 60px;
    //  text-shadow: 5px 5px 5px white;
    //}}

    @-webkit-keyframes emojis-fall {{
    0% {{
        top: -10%;
    }}
    100% {{
        top: 100%;
    }}
    }}
    @-webkit-keyframes emojis-shake {{
    0% {{
        -webkit-transform: translateX(0px);
        transform: translateX(0px);
    }}
    50% {{
        -webkit-transform: translateX(20px);
        transform: translateX(20px);
    }}
    100% {{
        -webkit-transform: translateX(0px);
        transform: translateX(0px);
    }}
    }}
    @keyframes emojis-fall {{
    0% {{
        top: -10%;
    }}
    100% {{
        top: 100%;
    }}
    }}
    @keyframes emojis-shake {{
    0% {{
        transform: translateX(0px);
    }}
    25% {{
        transform: translateX(15px);
    }}
    50% {{
        transform: translateX(-15px);
    }}
    100% {{
        transform: translateX(0px);
    }}
    }}

    .emoji {{
    position: fixed;
    top: -10%;
    z-index: 99999;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
    cursor: default;
    -webkit-animation-name: emojis-fall, emojis-shake;
    -webkit-animation-duration: 5s, 3s;
    -webkit-animation-timing-function: linear, ease-in-out;
    -webkit-animation-iteration-count: {animation_length}, {animation_length}; // overall length
    -webkit-animation-play-state: running, running;
    animation-name: emojis-fall, emojis-shake;
    animation-duration: {falling_speed}s, 3s;  // fall speed
    animation-timing-function: linear, ease-in-out;
    animation-iteration-count: {animation_length}, {animation_length}; // overall length
    animation-play-state: running, running;
    }}
    .emoji:nth-of-type(0) {{
    left: 1%;
    -webkit-animation-delay: 0s, 0s;
    animation-delay: 0s, 0s;
    }}
    .emoji:nth-of-type(1) {{
    left: 10%;
    -webkit-animation-delay: 1s, 1s;
    animation-delay: 1s, 1s;
    }}
    .emoji:nth-of-type(2) {{
    left: 20%;
    -webkit-animation-delay: 6s, 0.5s;
    animation-delay: 6s, 0.5s;
    }}
    .emoji:nth-of-type(3) {{
    left: 30%;
    -webkit-animation-delay: 4s, 2s;
    animation-delay: 4s, 2s;
    }}
    .emoji:nth-of-type(4) {{
    left: 40%;
    -webkit-animation-delay: 2s, 2s;
    animation-delay: 2s, 2s;
    }}
    .emoji:nth-of-type(5) {{
    left: 50%;
    -webkit-animation-delay: 8s, 3s;
    animation-delay: 8s, 3s;
    }}
    .emoji:nth-of-type(6) {{
    left: 60%;
    -webkit-animation-delay: 6s, 2s;
    animation-delay: 6s, 2s;
    }}
    .emoji:nth-of-type(7) {{
    left: 70%;
    -webkit-animation-delay: 2.5s, 1s;
    animation-delay: 2.5s, 1s;
    }}
    .emoji:nth-of-type(8) {{
    left: 80%;
    -webkit-animation-delay: 1s, 0s;
    animation-delay: 1s, 0s;
    }}
    .emoji:nth-of-type(9) {{
    left: 90%;
    -webkit-animation-delay: 3s, 1.5s;
    animation-delay: 3s, 1.5s;
    }}

    </style>
    """,
        unsafe_allow_html=True,
    )

    st.write(
        f"""
    <!--get emojis from https://getemoji.com-->
    <div class="emojis">
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

rain(emoji='💷',
     font_size=20,
    falling_speed=50,
    animation_length="infinite")
