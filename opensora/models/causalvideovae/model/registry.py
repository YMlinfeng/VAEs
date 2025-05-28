# 详细介绍详见：https://www.yuque.com/yumulinfengfirepigreturn/rhlx3q/qvsvv8wxsmzxehcb/edit?toc_node_uuid=QEEV3aQoEw5qJ4qm#d8d1091c
class ModelRegistry:
    _models = {} # # 一个用来保存“模型名称 -> 模型类”的字典

    @classmethod # 表示这个方法是类方法，用类名就能调用（不需要创建对象）；会把类本身 cls 作为第一个参数传入
    def register(cls, model_name):
        def decorator(model_class):
            cls._models[model_name] = model_class
            return model_class
        return decorator

    @classmethod
    def get_model(cls, model_name):
        return cls._models.get(model_name) # # 根据名字取模型类
    
'''
也就是说，
@ModelRegistry.register("WFVAE")
class WFVAEModel:
等价于
WFVAEModel = ModelRegistry.register("WFVAE")(WFVAEModel)
等价于
WFVAEModel = decorator(WFVAEModel)
好的，那接下来怎么理解函数运行的过程？
执行过程：

-把 WFVAEModel 作为 model_class 传入
-执行注册：ModelRegistry._models["WFVAE"] = WFVAEModel
-返回原封不动的 WFVAEModel 类

整个过程就相当于给WFVAEModel类起了个别名叫WFVAE

'''