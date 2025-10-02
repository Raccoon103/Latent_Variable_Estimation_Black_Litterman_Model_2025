from datetime import datetime


DATA_START_DATE = '1991-01-01'
DECISION_START_DATE = '1994-01-01'
END_DATE = '2024-02-24'

# Technical indicators used
NUM_INDICATORS = 9
CACHE_DIR = "data/yf_cache"

# Benchmarks
BENCHMARKS = ['^DJI', 'SPY']

# Historical components of the Dow Jones Industrial Average
DOW_JONES_COMPONENTS = {
    "1991-05-06": ["AA", "AXP", "BA", "CAT", "CVX", "DD", "DIS", "FL", "GE", "GT", "HON", "IBM", "IP", "JPM", "KO", "MCD", "MMM", "MO", "MRK", "PG", "T", "XOM"],
    "1997-03-17": ["AA", "AXP", "BA", "CAT", "CVX", "DD", "DIS", "GE", "GT", "HON", "HPQ", "IBM", "IP", "JNJ", "JPM", "KO", "MCD", "MMM", "MO", "MRK", "PG", "T", "TRV", "WMT", "XOM"],
    "1999-11-01": ["AA", "AXP", "BA", "C", "CAT", "DD", "DIS", "GE", "HD", "HON", "HPQ", "IBM", "INTC", "IP", "JNJ", "JPM", "KO", "MCD", "MMM", "MO", "MRK", "MSFT", "PG", "T", "WMT", "XOM"],
    "2004-04-08": ["AA", "AIG", "AXP", "BA", "C", "CAT", "DD", "DIS", "GE", "HD", "HON", 'HPQ', "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MO", "MRK", "MSFT", "PFE", "PG", "T", "VZ", "WMT", "XOM"],
    "2008-02-19": ["AA", "AIG", "AXP", "BA", "BAC", "C", "CAT", "CVX", "DD", "DIS", "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "PFE", "PG", "T", "VZ", "WMT", "XOM"],
    "2008-09-22": ["AA", "AXP", "BA", "BAC", "C", "CAT", "CVX", "DD", "DIS", "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "PFE", "PG", "T", "VZ", "WMT", "XOM"],
    "2009-06-08": ["AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "PFE", "PG", "T", "TRV", "VZ", "WMT", "XOM"],
    "2012-09-24": ["AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "PFE", "PG", "T", "TRV", "UNH", "VZ", "WMT", "XOM"],
    "2013-09-23": ["AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "T", "TRV", "UNH", "VZ", "WMT", "XOM"],
    "2015-03-19": ["AXP", "AAPL", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "VZ", "WMT", "XOM"],
    "2018-06-26": ["AXP", "AAPL", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "VZ", "WBA", "WMT", "XOM"],
    "2019-04-02": ["AXP", "AAPL", "BA", "CAT", "CSCO", "CVX", "DIS", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "VZ", "WBA", "WMT", "XOM"],
    "2020-08-31": ["AXP", "AAPL", "AMGN", "BA", "CAT", "CSCO", "CVX", "DIS", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "VZ", "WBA", "WMT"],
    "Union": ['AA', 'AIG', 'AAPL', 'AMGN', 'AXP', 'BA', 'BAC', 'C', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', "FL", 'GE', 'GS', "GT", 'HD', 'HON', 'HPQ', 'IBM', 'INTC', 'IP',
              'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MO', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'T', 'TRV', 'UNH', 'VZ', 'WBA', 'WMT', 'XOM'],
    "Removed" : ["BS", 'CRM', 'DOW', 'DWDP', "EK", 'GM', "KHC", 'S', 'SBC', "TX", 'UK', 'UTX', 'V', "WX"],
}

SPDR_COMPONENTS = {
    "1998-12-22": ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
    #add "XLRE"
    "2015-10-08": ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE'],
    #add "XLC"
    "2018-06-19": ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE'],
    
    # Add future changes here...
    "Union": ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLC', 'XLRE'],
    "Removed" : [],
}

MSCI_WORLD_INDEX_COMPONENTS = {
    # Components with the start dates of their respective ETFs
    "1996-03-18": ['EWA', 'EWC', 'EWD', 'EWG', 'EWH', 'EWI', 'EWJ', 'EWK', 
                   'EWL', 'EWN', 'EWO', 'EWP', 'EWQ', 'EWS', 'EWU'],  # Initial ETFs launched
    "2008-03-28": ['EWA', 'EWC', 'EWD', 'EWG', 'EWH', 'EWI', 'EWJ', 'EWK', 
                   'EWL', 'EWN', 'EWO', 'EWP', 'EWQ', 'EWS', 'EWU', 'EIS'],  # Adding Israel
    "2010-05-07": ['EWA', 'EWC', 'EWD', 'EWG', 'EWH', 'EWI', 'EWJ', 'EWK', 
                   'EWL', 'EWN', 'EWO', 'EWP', 'EWQ', 'EWS', 'EWU', 'EIS', 'EUSA'],  # Adding USA
    "2010-09-02": ['EWA', 'EWC', 'EWD', 'EWG', 'EWH', 'EWI', 'EWJ', 'EWK', 
                   'EWL', 'EWN', 'EWO', 'EWP', 'EWQ', 'EWS', 'EWU', 'EIS', 'EUSA', 'ENZL'],  # Adding New Zealand
    "2012-01-24": ['EWA', 'EWC', 'EWD', 'EWG', 'EWH', 'EWI', 'EWJ', 'EWK', 
                   'EWL', 'EWN', 'EWO', 'EWP', 'EWQ', 'EWS', 'EWU', 'EIS', 'EUSA', 'ENZL', 'ENOR'],  # Adding Norway
    "2012-01-26": ['EWA', 'EWC', 'EWD', 'EWG', 'EWH', 'EWI', 'EWJ', 'EWK', 
                   'EWL', 'EWN', 'EWO', 'EWP', 'EWQ', 'EWS', 'EWU', 'EIS', 'EUSA', 'ENZL', 'ENOR', 'EFNL', 'EDEN'],  # Adding Finland and Denmark
    "2013-11-12": ['EWA', 'EWC', 'EWD', 'EWG', 'EWH', 'EWI', 'EWJ', 'EWK', 
                   'EWL', 'EWN', 'EWO', 'EWP', 'EWQ', 'EWS', 'EWU', 'EIS', 'EUSA', 'ENZL', 'ENOR', 'EFNL', 'EDEN', 'PGAL'],  # Adding Portugal
    # Union: all ETFs that have ever been part of the MSCI World Index
    "Union": ['EWA', 'EWC', 'EWD', 'EWG', 'EWH', 'EWI', 'EWJ', 'EWK', 
              'EWL', 'EWN', 'EWO', 'EWP', 'EWQ', 'EWS', 'EWU', 'EIS', 'EUSA', 
              'ENZL', 'ENOR', 'EFNL', 'EDEN', 'PGAL'],
    # Removed: countries or regions without ETFs
    "Removed": ['Greece'],  # Greece was initially listed in developed country index, but then emerging market index.
}


# only need to set this
COMPONENT = MSCI_WORLD_INDEX_COMPONENTS

# Automatically derive component name from the component variable
_COMPONENT_NAME_MAP = {
    id(DOW_JONES_COMPONENTS): "DJ",
    id(SPDR_COMPONENTS): "SPDR", 
    id(MSCI_WORLD_INDEX_COMPONENTS): "MSCI_World_Index"
}

COMPONENT_NAME = _COMPONENT_NAME_MAP[id(COMPONENT)]

