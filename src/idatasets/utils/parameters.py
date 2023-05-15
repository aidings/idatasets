import inspect
from typing import Dict

def comkey(module, kwargs:Dict):
	if isinstance(module, list):
		mkeys = set(module)
	else:
		mkeys = set(inspect.signature(module).parameters.keys())
	ikeys = set(kwargs.keys())	

	keys = mkeys & ikeys

	param = {}
	for key in keys:
		param[key] = kwargs.get(key)
	
	return param

