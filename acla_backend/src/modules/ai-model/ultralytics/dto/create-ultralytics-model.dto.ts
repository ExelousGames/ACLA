import { BadRequestException } from '@nestjs/common';
import {
  ULTRALYTICS_TASKS,
  UltralyticsTask,
} from '../../../../schemas/ultralytics-model.schema';

export interface CreateUltralyticsModelDto {
  name: string;
  task: UltralyticsTask;
  classNames: string[];
  metadata?: Record<string, unknown>;
}

export interface UltralyticsModelUpload {
  path: string;
  originalname: string;
  size: number;
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

// Multipart fields arrive as strings; this backend has no global validation pipe.
export function parseUltralyticsModelMetadata(
  raw: unknown,
): CreateUltralyticsModelDto {
  if (typeof raw !== 'string') {
    throw new BadRequestException(
      'metadata must be a JSON object encoded as a string',
    );
  }

  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new BadRequestException('metadata must contain valid JSON');
  }
  if (!isObject(value)) {
    throw new BadRequestException('metadata must be a JSON object');
  }
  if (
    typeof value.name !== 'string' ||
    !value.name.trim() ||
    value.name.length > 200
  ) {
    throw new BadRequestException(
      'name must be a non-empty string of at most 200 characters',
    );
  }
  if (!ULTRALYTICS_TASKS.includes(value.task as UltralyticsTask)) {
    throw new BadRequestException(
      `task must be one of: ${ULTRALYTICS_TASKS.join(', ')}`,
    );
  }
  if (
    !Array.isArray(value.classNames) ||
    value.classNames.length === 0 ||
    !value.classNames.every(
      (name: unknown) => typeof name === 'string' && name.trim().length > 0,
    ) ||
    new Set(value.classNames).size !== value.classNames.length
  ) {
    throw new BadRequestException(
      'classNames must be a non-empty array of unique class names in class-ID order',
    );
  }
  if (value.metadata !== undefined && !isObject(value.metadata)) {
    throw new BadRequestException('metadata.metadata must be a JSON object');
  }

  return {
    name: value.name.trim(),
    task: value.task as UltralyticsTask,
    classNames: value.classNames as string[],
    metadata: value.metadata,
  };
}
