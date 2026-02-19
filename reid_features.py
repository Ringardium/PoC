import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from ultralytics import YOLO
from PIL import Image
import logging
from efficientnet_pytorch import EfficientNet

class ColorHistogramReID:
    """
    CV2 히스토그램을 이용한 색상 분포 기반 Re-ID
    """
    
    def __init__(self, bins = 32):
        self.bins = bins
        self.feature_dim = bins * 3  # RGB 각 채널당 bins개씩
        
    def extract_features(self, crop_image):
        """
        이미지 크롭에서 색상 히스토그램 특징 추출
        
        Args:
            crop_image: numpy array (H, W, 3) BGR 이미지
        
        Returns:
            features: numpy array (feature_dim,) 정규화된 특징 벡터
        """
        try:
            if crop_image.size == 0:
                return np.random.rand(self.feature_dim).astype(np.float32)
            
            # 이미지 크기 정규화
            crop_resized = cv2.resize(crop_image, (64, 128))
            
            # BGR을 RGB로 변환
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            
            features = []
            
            # 각 RGB 채널별로 히스토그램 계산
            for channel in range(3):
                hist = cv2.calcHist([crop_rgb], [channel], None, [self.bins], [0, 256])
                hist = hist.flatten()
                features.extend(hist)
            
            features = np.array(features, dtype=np.float32)
            
            # L2 정규화
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            logging.warning(f"색상 히스토그램 특징 추출 실패: {e}")
            return np.random.rand(self.feature_dim).astype(np.float32)


class OSNetReID:
    """
    TorchReID OSNet을 이용한 딥러닝 기반 Re-ID
    """
    
    def __init__(self, model_name='osnet_x1_0', pretrained=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.feature_dim = 512  # OSNet 기본 특징 차원
        
        # 데이터 전처리 변환
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 모델 초기화
        self.model = None
        self.pretrained = pretrained
        self._load_model()
        
    def _load_model(self):
        """
        OSNet 모델 로드 
        """
        try:
            import torchreid
            
            # TorchReID OSNet 모델 로드
            self.model = torchreid.models.build_model(
                name = self.model_name,
                num_classes=1000,  # 임시값 (실제 사용시에는 feature extraction만)
                pretrained = self.pretrained
            )
            
            # Feature extraction 모드로 설정
            self.model.eval()
            self.model = self.model.to(self.device)
            
            logging.info(f"TorchReID {self.model_name} 모델 로드 완료")
            
        except ImportError:
            logging.warning("TorchReID 라이브러리가 없습니다")
            
        except Exception as e:
            logging.warning(f"OSNet 모델 로드 실패 : {e}")

    def extract_features(self, crop_image):
        """
        이미지 크롭에서 딥러닝 특징 추출
        
        Args:
            crop_image: numpy array (H, W, 3) BGR 이미지
        
        Returns:
            features: numpy array (feature_dim,) 정규화된 특징 벡터
        """
        try:
            if crop_image.size == 0:
                return np.random.rand(self.feature_dim).astype(np.float32)
            
            # BGR을 RGB로 변환
            crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            
            # PIL Image로 변환
            pil_image = Image.fromarray(crop_rgb)
            
            # 전처리 적용
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # 특징 추출
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.cpu().numpy().flatten()
            
            # L2 정규화
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features.astype(np.float32)
            
        except Exception as e:
            logging.warning(f"OSNet 특징 추출 실패: {e}")
            return np.random.rand(self.feature_dim).astype(np.float32)

class EfficientNetReID:
    """
    EfficientNet을 이용한 딥러닝 기반 Re-ID
    """
    
    def __init__(self, model_name='efficientnet-b0', pretrained=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.feature_dim = 512
        
        # 데이터 전처리 변환
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 모델 초기화
        self.model = model_name
        self.pretrained = pretrained
        self._load_model()
        
    def _load_model(self):
        """
        EfficientNet 모델 로드
        """
        try:
            import torchreid
            
            # TorchReID OSNet 모델 로드
            self.model = EfficientNet.from_pretrained(self.model_name)
            
            # Feature extraction 모드로 설정
            self.model.eval()
            self.model = self.model.to(self.device)
            
            logging.info(f"TorchReID {self.model_name} 모델 로드 완료")
            
        except ImportError:
            logging.warning("EfficientNet이 없습니다.")
            
        except Exception as e:
            logging.warning(f"EfficientNet 모델 로드 실패 : {e}")

    def extract_features(self, crop_image):
        """
        이미지 크롭에서 딥러닝 특징 추출
        
        Args:
            crop_image: numpy array (H, W, 3) BGR 이미지
        
        Returns:
            features: numpy array (feature_dim,) 정규화된 특징 벡터
        """
        try:
            if crop_image.size == 0:
                return np.random.rand(self.feature_dim).astype(np.float32)
            
            # BGR을 RGB로 변환
            crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            
            # PIL Image로 변환
            pil_image = Image.fromarray(crop_rgb)
            
            # 전처리 적용
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # 특징 추출
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.cpu().numpy().flatten()
            
            # L2 정규화
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features.astype(np.float32)
            
        except Exception as e:
            logging.warning(f"EfficeintNet 특징 추출 실패: {e}")
            return np.random.rand(self.feature_dim).astype(np.float32)


class HybridReID:
    """
    색상 히스토그램과 OSNet을 결합한 하이브리드 Re-ID
    """
    
    def __init__(self, use_osnet=True, use_effnet= True, osnet_weight=0.45, effnet_weight = 0.45 ):
        self.use_hist = True
        self.use_osnet = use_osnet
        self.use_effnet = use_effnet
        self.osnet_weight = osnet_weight
        self.effnet_weight = effnet_weight
        
        # 색상 히스토그램 특징 추출기
        self.color_extractor = ColorHistogramReID()
        
        # OSNet 특징 추출기(default:사용)
        self.osnet_extractor = None
        # EfficientNet 특징 추출기(default:사용안함)
        self.effnet_extractor = None

        if use_osnet :
            try:
                self.osnet_extractor = OSNetReID()
            except:
                logging.warning("OSNet 초기화 실패")
                osnet_weight = 0.0
        
        if use_effnet : 
            try:
                self.osnet_extractor = EfficientNetReID()
            except:
                logging.warning("EfficientNet 초기화 실패")
                effnet_weight = 0.0

        self.hist_weight = 1.0 - osnet_weight - effnet_weight
        logging.warning(f"\nHistgram 색상 분포 특징 : {self.hist_weight}\nOSnet 특징 : {self.osnet_weight}\nEfficientNet 특징 : {self.effnet_weight}")
        
        if self.use_osnet and self.use_effnet:
            try:
                self.osnet_extractor = OSNetReID()
                self.effnet_extractor = EfficientNetReID()
                self.feature_dim = self.color_extractor.feature_dim + self.osnet_extractor.feature_dim + self.effnet_extractor.feature_dim
            except:
                logging.warning("OSNet, EfficientNet 초기화 실패")
                effnet_weight = 0.0
                osnet_weight = 0.0
        elif self.use_osnet and not self.use_effnet:
            self.feature_dim = self.color_extractor.feature_dim + self.osnet_extractor.feature_dim    
        elif self.use_effnet and not self.use_osnet:
            self.feature_dim = self.color_extractor.feature_dim + self.effnet_extractor.feature_dim
        else:
            self.feature_dim = self.color_extractor.feature_dim
             


    
    def extract_features(self, crop_image):
        """
        하이브리드 특징 추출 (색상 + 딥러닝)
        
        Args:
            crop_image: numpy array (H, W, 3) BGR 이미지
        
        Returns:
            features: numpy array (feature_dim,) 정규화된 특징 벡터
        """
        #try:
        # 색상 히스토그램 특징
        hist_features = self.color_extractor.extract_features(crop_image)
        
        if (self.use_osnet and self.use_effnet):
            osnet_features = self.osnet_extractor.extract_features(crop_image)
            effnet_features = self.effnet_extractor.extract_features(crop_image)
            
            # 가중 결합
            hist_features_weighted = hist_features * self.hist_weight
            osnet_features_weighted = osnet_features * self.osnet_weight
            effnet_features_weighted = effnet_features * self.effnet_weight
            
            # 특징 연결
            combined_features = np.concatenate([hist_features_weighted, osnet_features_weighted])
            combined_features = np.concatenate([combined_features, effnet_features_weighted])
            
            # 최종 정규화
            norm = np.linalg.norm(combined_features)
            if norm > 0:
                combined_features = combined_features / norm
            
            return combined_features.astype(np.float32)
            
        elif self.use_osnet and self.osnet_extractor is not None:
            # OSNet 특징
            osnet_features = self.osnet_extractor.extract_features(crop_image)

            # EfficientNet 특징
            
            # 가중 결합
            hist_features_weighted = hist_features * self.hist_weight
            osnet_features_weighted = osnet_features * self.osnet_weight

            # 특징 연결
            combined_features = np.concatenate([hist_features_weighted, osnet_features_weighted])
            
            # 최종 정규화
            norm = np.linalg.norm(combined_features)
            if norm > 0:
                combined_features = combined_features / norm
            
            return combined_features.astype(np.float32)
        
        
        elif self.use_effnet and self.effnet_extractor is not None :
            # EfficientNet 특징
            effnet_features = self.effnet_extractor.extract_features(crop_image)
            
            # 가중 결합
            hist_features_weighted = hist_features * self.hist_weight
            effnet_features_weighted = effnet_features * self.effnet_weight
            
            # 특징 연결
            combined_features = np.concatenate([hist_features_weighted, effnet_features_weighted])
            
            # 최종 정규화
            norm = np.linalg.norm(combined_features)
            if norm > 0:
                combined_features = combined_features / norm
            
            return combined_features.astype(np.float32)
            
        else:
            return hist_features
        """        
        except Exception as e:
            logging.warning(f"하이브리드 특징 추출 실패: {e}")"""


def create_reid_extractor(method='hybrid', **kwargs):
    """
    Re-ID 특징 추출기 팩토리 함수
    
    Args:
        method: 'histogram', 'osnet', 'hybrid' 중 하나
        **kwargs: 각 방법별 추가 매개변수
    
    Returns:
        특징 추출기 객체
    """
    if method == 'histogram':
        return ColorHistogramReID(**kwargs)
    elif method == 'osnet':
        return OSNetReID(**kwargs)
    elif method == 'efficientnet':
        return EfficientNet(**kwargs)
    elif method == 'hybrid':
        return HybridReID(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 Re-ID 방법: {method}")


if __name__ == "__main__":
    # 테스트 코드
    import time
    
    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
    
    # 각 방법별 테스트
    methods = {
        'histogram': ColorHistogramReID(),
        'osnet': OSNetReID(),
        'effnet' : EfficientNetReID(),
        'hybrid': HybridReID()
    }
    
    for name, extractor in methods.items():
        start_time = time.time()
        features = extractor.extract_features(test_image)
        end_time = time.time()
        
        print(f"{name}: 특징 차원 {len(features)}, 처리 시간 {end_time-start_time:.4f}초")