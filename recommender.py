%%writefile recommender.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

@st.cache(allow_output_mutation=True)
def persistlistens():
    return {}

def sparse_list(ratings, artist_ids):
  output = [0] * len(artist_ids)
  not_found = []
  for k, v in ratings.items():
    try:
      output[k] = v
    except ValueError:
      not_found.append(k)

  return output, not_found

@st.experimental_memo
def retrieve_item_dict():
  with open('drive/MyDrive/CA4015/artists.pickle', 'rb') as p:
    item_dict = pickle.load(p)
  return item_dict

@st.experimental_memo
def retrieve_artist_embeddings():
  with open('drive/MyDrive/CA4015/embeddings.pickle', 'rb') as p:
    embeddings = pickle.load(p)
  return embeddings

@st.experimental_memo
def compute_recommendations(ratings, artist_id_name):
  embeddings = retrieve_artist_embeddings()
  
  rating_vector, not_found = sparse_list(ratings, artist_id_name.keys())
  ratings = np.asarray(rating_vector)

  latent_rep = np.matmul(ratings, embeddings)
  recommendations = np.matmul(latent_rep, np.transpose(embeddings))
  names = [artist_id_name[artist_id] for artist_id in artist_id_name.keys()]

  return zip(names, recommendations), not_found


def main():

  st.set_page_config(
    page_title="User Recommendations",
    layout="wide"
)

  st.header("User Recommendations")

  artist_id_name = retrieve_item_dict()
  artist_name_id = {v: k for k, v in artist_id_name.items()}
  artist_names = artist_id_name.values()

  user_listens = persistlistens()

  if "recommendations" not in locals():
    recommendations = None

  left, right = st.columns([2, 5])

  left.subheader("User input")

  right.subheader("Your recommentations can be seen here once submitted")
  
  selected_artists = left.multiselect("Select artist to rate", artist_names)
  
  expander =  left.expander("Rate your selected artists", expanded=False)

  user_listens.clear()
  for artist in selected_artists:

    rating = expander.selectbox(f"Rating for {artist}", [0,1,2,3,4,5])

    id = artist_name_id[artist]
    user_listens[id] = rating
  
  if len(user_listens) > 0:
    output, not_found = compute_recommendations(user_listens, artist_id_name)
    right.dataframe(pd.DataFrame(output, columns=["Items", "Predicted relevance"]).sort_values(by=["Predicted relevance"], ascending=False))


if __name__ == '__main__':
	main()
