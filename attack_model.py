# attack_model.py
import torch

class PoisonFRS:
    def __init__(self, model, item_embedding, target_item, popular_items, filler_items, scale_factor):
        self.model = model
        self.item_embedding = item_embedding
        self.target_item = target_item
        self.popular_items = popular_items
        self.filler_items = filler_items
        self.scale_factor = scale_factor

    def craft_updates(self, global_round):
        # Craft model updates for fake users
        updates = {}
        for item, embedding in self.item_embedding.items():
            if item == self.target_item:
                # Scale the target item embedding
                updates[item] = self.scale_factor * embedding
            elif item in self.popular_items:
                # Keep popular items as they are
                updates[item] = embedding
            elif item in self.filler_items:
                # Keep filler items as they are
                updates[item] = embedding
            else:
                # Null update for other items
                updates[item] = torch.zeros_like(embedding)
        return updates

    def apply_update(self, updates):
        # Apply the crafted updates to the model
        new_item_embeddings = self.model.item_embeddings.data.clone()  # 创建当前item_embeddings的副本
        for item, update in updates.items():
            if item < new_item_embeddings.size(0):  # 确保item索引在范围内
                new_item_embeddings[item] = update  # 更新副本中的值
        self.model.item_embeddings.data = new_item_embeddings  # 将更新后的副本赋值回item_embeddings