import React, { useRef, useState } from 'react';
import { DropdownMenu, Button, Dialog, Flex, Text } from '@radix-ui/themes';
import { GearIcon, UploadIcon } from '@radix-ui/react-icons';
import apiService from 'services/api.service';

interface SettingsMenuProps {
    mapName: string;
    onImageUploaded?: () => void;
}

const SettingsMenu: React.FC<SettingsMenuProps> = ({ mapName, onImageUploaded }) => {
    const [isUploadDialogOpen, setIsUploadDialogOpen] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadMessage, setUploadMessage] = useState('');
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileSelect = () => {
        fileInputRef.current?.click();
    };

    // user selected a image in the dialog, this handles image upload
    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {

        // Get the selected file
        const file = event.target.files?.[0];

        // Check if a file was selected
        if (!file) return;

        // Validate file type - only allow JPG and PNG
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!allowedTypes.includes(file.type)) {
            setUploadMessage('Please select a JPG or PNG image file');
            return;
        }

        // Show uploading state
        setIsUploading(true);
        setUploadMessage('');

        try {
            const formData = new FormData();
            formData.append('image', file);
            formData.append('mapName', mapName);

            // Use apiService for file upload
            const response = await apiService.uploadFile<{ success: boolean; message: string }>('/racingmap/map/upload-image', formData);

            // Handle response
            if (response.data.success) {
                setUploadMessage('Image uploaded successfully!');
                onImageUploaded?.();
                setTimeout(() => {
                    setIsUploadDialogOpen(false);
                    setUploadMessage('');
                }, 2000);
            } else {
                setUploadMessage(response.data.message || 'Upload failed');
            }
        } catch (error: any) {
            console.error('Upload error:', error);
            if (error.response?.status === 401) {
                setUploadMessage('Authentication failed. Please log in again.');
            } else if (error.response?.status === 404) {
                setUploadMessage('Map not found');
            } else {
                setUploadMessage(error.response?.data?.message || error.message || 'Upload failed');
            }
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <>
            <DropdownMenu.Root>
                <DropdownMenu.Trigger>
                    <Button
                        variant="soft"
                        size="2"
                        style={{
                            position: 'absolute',
                            top: '10px',
                            left: '10px',
                            zIndex: 1000,
                        }}
                    >
                        <GearIcon />
                        Settings
                    </Button>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content>
                    <DropdownMenu.Item onClick={() => setIsUploadDialogOpen(true)}>
                        <UploadIcon />
                        Upload Map Image
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            <Dialog.Root open={isUploadDialogOpen} onOpenChange={setIsUploadDialogOpen}>
                <Dialog.Content style={{ maxWidth: 450 }}>
                    <Dialog.Title>Upload Map Image</Dialog.Title>
                    <Dialog.Description size="2" mb="4">
                        Upload a new background image for the map "{mapName}" (JPG or PNG format only)
                    </Dialog.Description>

                    <Flex direction="column" gap="3">
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                            style={{ display: 'none' }}
                        />

                        <Button
                            onClick={handleFileSelect}
                            disabled={isUploading}
                            variant="outline"
                            size="2"
                        >
                            <UploadIcon />
                            {isUploading ? 'Uploading...' : 'Select Image'}
                        </Button>

                        {uploadMessage && (
                            <Text
                                size="2"
                                color={uploadMessage.includes('successfully') ? 'green' : 'red'}
                            >
                                {uploadMessage}
                            </Text>
                        )}
                    </Flex>

                    <Flex gap="3" mt="4" justify="end">
                        <Dialog.Close>
                            <Button variant="soft" color="gray">
                                Close
                            </Button>
                        </Dialog.Close>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>
        </>
    );
};

export default SettingsMenu;
