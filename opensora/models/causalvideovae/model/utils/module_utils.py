import importlib # importlib 是 Python 的标准库，提供运行时导入模块的能力

Module = str
MODULES_BASE = "opensora.models.causalvideovae.model.modules."

def resolve_str_to_obj(str_val, append=True):
    if append:
        str_val = MODULES_BASE + str_val
    module_name, class_name = str_val.rsplit('.', 1) # rsplit() 是 split() 的变体，从右侧开始分割；返回的是一个 list，可以直接使用多变量解包（如 a, b = ...）
    module = importlib.import_module(module_name) # 返回的是一个 模块对象（等价于 import opensora.models.my_module as module）
    return getattr(module, class_name)

def create_instance(module_class_str: str, **kwargs):
    module_name, class_name = module_class_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**kwargs)


'''
③ getattr(module, class_name)
return getattr(module, class_name)
getattr() 是 Python 的内置函数；
用于从一个对象中获取属性或方法；
等价于：module.class_name，但 class_name 是字符串时必须用 getattr。
✅ 语法知识点：

getattr(obj, "attr_name") ≈ obj.attr_name；
常用于需要动态访问属性的情况（如反射、插件机制等）。


'''