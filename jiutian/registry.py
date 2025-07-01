class Registry:
    
    root = ""
    
    mapping = {
        "builder_name_mapping": {},
        "evaluator_name_mapping": {},
    }
    
    @classmethod
    def register_builder(cls, name):
        r"""Register a dataset builder to registry with key 'name'

        Args:
            name: Key with which the dataset builder will be registered.

        Usage:

            from .common.registry import registry
        """

        def wrap(builder_func):
            if name in cls.mapping["builder_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["builder_name_mapping"][name]
                    )
                )
            cls.mapping["builder_name_mapping"][name] = builder_func
            return builder_func

        return wrap

    @classmethod
    def register_evaluator(cls, name):
        r"""Register a task evaluator to registry with key 'name'

        Args:
            name: Key with which the task evaluator will be registered.

        Usage:

            from .common.registry import registry
        """

        def wrap(eval_func):
            if name in cls.mapping["evaluator_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["evaluator_name_mapping"][name]
                    )
                )
            cls.mapping["evaluator_name_mapping"][name] = eval_func
            return eval_func

        return wrap

    @classmethod
    def get_builder_func(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)
    
    @classmethod
    def get_evaluator_func(cls, name):
        return cls.mapping["evaluator_name_mapping"].get(name, None)

    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping["builder_name_mapping"].keys())

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from mmf.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()
