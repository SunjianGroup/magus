import re

def camel2snake(name):
    snake_case = re.sub(r"(?P<key>[A-Z])", r"_\g<key>", name)
    return snake_case.lower().strip('_')

def snake2camel(name):
    return re.sub(r"(_[a-z])", lambda x: x.group(1)[1].upper(), name)

#seperate "key           //note" format
def seperate_notes_from_name(keyname):
    note_start = keyname.find("//")
    if note_start == -1:
        return keyname.rstrip(), keyname.rstrip()
    else:
        return keyname[:note_start].rstrip(), keyname[note_start+2:].rstrip()

#requirement/default with notes -> {clean key: note}
def refine_help_info(item):
    info_ = {}
    if type(item) is list:
        for name in item:
            key, help = seperate_notes_from_name(name)
            info_[key] = [help]
    elif type(item) is dict:
        for name in item:
            key, help = seperate_notes_from_name(name)
            value = item[name]    
            info_[key] = [help, value]
    return info_

import abc
#This is a base class to set parameters for magus.
class Parmbase(abc.ABC):
    __requirement = NotImplemented
    __default = NotImplemented

    @staticmethod
    def transform(item):
        refined = refine_help_info(item)
        if type(item) is list:
            return list(refined.keys())
        else:
            return {key: refined[key][-1] for key in refined}

    @staticmethod
    def check_parameters(instance, parameters, Requirement, Default):
        for key in Requirement:
            if key in parameters:
                setattr(instance, key, parameters[key])
            elif snake2camel(key) in parameters:
                setattr(instance, key, parameters[snake2camel(key)])
            else:
                raise Exception("'{}' must have {}".format(instance, key))

        for key in Default.keys():
            if key in parameters:
                setattr(instance, key, parameters[key])
            elif snake2camel(key) in parameters:
                setattr(instance, key, parameters[snake2camel(key)])  
            else:
                setattr(instance, key, Default[key])

    @classmethod
    def self_help_info(cls):
        all_info = {}
        key = "_{}__help_list".format(cls.__name__)
        hl = cls.__dict__[key] if key in cls.__dict__ else ['default', 'requirement']
        for name in hl:
            item = cls.__dict__["_{}__{}".format(cls.__name__, name)]
            all_info.update({name:  refine_help_info(item)})
        return cls, all_info

    @classmethod
    def export_all_help_info(cls):
        all_info = []
        now_cls = cls
        while not (now_cls is Parmbase):
           all_info.append(now_cls.self_help_info())
           now_cls = now_cls.__base__
        
        return all_info
