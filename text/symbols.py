from text import cmudict

_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'abcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters)  # + _arpabet
symbols_ = ['\\_'] + list(_special) + list(_punctuation) + list(_letters)

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols_)}