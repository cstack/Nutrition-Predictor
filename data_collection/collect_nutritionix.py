from nutritionix import Nutritionix
import private_consts

nix = Nutritionix(app_id=private_consts.NUTRITIONIX["app_id"],
  api_key=private_consts.NUTRITIONIX["api_key"])

# Example code
pizza = nix.search("pizza")
results = pizza.json()
print results
"""
results looks like:
{u'total_hits': 12612,
u'hits': [
  {
    u'_score': 3.550068,
    u'_type': u'item',
    u'_id': u'5266a2359f05a39eb300fc90',
    ...
  },
  ...
],
u'max_score': 3.550068}
"""
