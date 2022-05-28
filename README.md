# House-Price-Prediction-Paris-Flask

The dataset that the model was trained on is the [Paris Housing Price Prediction](https://www.kaggle.com/datasets/mssmartypants/paris-housing-price-prediction)

A linear regression model that predicts the price of a house in Paris using the following features:
- squareMeters
- numberOfRooms
- hasYard
- hasPool
- floors (number of floors)
- cityCode (zip code)
- cityPartRange (the higher the range, the more exclusive the neighbourhood is)
- numPrevOwners (number of previous owners)
- made (year)
- isNewBuilt
- hasStormProtector
- basement (basement square meters)
- attic (attic square meteres)
- garage  (garage size)
- hasStorageRoom
- hasGuestRoom (number of guest rooms)
- price  (predicted value)


That model is then saved and deployed on flask to allow the user to enter their desired specification for the house.
