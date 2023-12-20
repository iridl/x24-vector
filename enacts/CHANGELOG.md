# ENACTS Maprooms

# Change Log for config files

## Backwards-incompatible changes

### `crop_suitability`

* `layers` are now called `map_text`

* each `map_text` entry has the suffix `_layer` dropped

### `flex_fcst`

* `flex_fcst` config value is now a list of dictionaries rather than a single dictionary.

## Nota Bene

### all maprooms

* It is no longer necessary to set maproom config keys to `null` in the config file to prevent them from appearing. Only maprooms that are explicitly configured in the config file will be created.

### `crop_suitability`

* `map_text` entries need not `id`

* `target_year` is dropped
